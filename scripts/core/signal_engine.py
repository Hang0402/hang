# -*- coding: utf-8 -*-
"""
量价关系信号引擎 —— 唯一策略核心
只使用价格和成交量，不引入任何其他指标（MACD/均线/布林带等）

信号分为两大类：
  A. 量价确认 —— 趋势延续信号
  B. 量价异常 —— 反转预警信号

每个信号返回：{signal_id, type, direction, confidence, description, suggestion}
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Signal:
    """统一信号结构"""
    signal_id: str          # 信号编号
    category: str           # 'confirm'(量价确认) / 'anomaly'(量价异常)
    direction: str          # 'bullish' / 'bearish' / 'neutral'
    confidence: float       # 0-100 置信度
    description: str        # 信号描述（情况）
    suggestion: str         # 对策建议
    date: str = ''          # 信号日期
    price: float = 0.0      # 当日收盘价
    volume_ratio: float = 1.0  # 量比
    extra: dict = field(default_factory=dict)


class VolumePriceEngine:
    """
    量价关系策略引擎

    策略哲学：
      成交量是价格的燃料。量价同步 = 趋势健康，量价背离 = 即将反转。
      核心公式：量比 = 今日成交量 / N日均量
               涨跌幅 = (今日收盘 - 昨日收盘) / 昨日收盘

    七大信号：
      【量价确认 - 趋势延续】
      S1. 量价齐升：价涨 + 放量(>1.3)  → 上涨趋势健康，持多/做多
      S2. 缩量回调：价跌 + 缩量(<0.7)  → 正常回调，趋势未坏

      【量价确认 - 趋势警示】
      S3. 价涨量缩：价涨 + 缩量(<0.7)  → 上涨乏力，警惕见顶
      S4. 价跌量增：价跌 + 放量(>1.3)  → 下跌加速，警惕深跌

      【量价异常 - 反转信号】
      S5. 二次探底缩量：价格逼近前低 + 量明显小于前低(<0.6)
                       → 底部确认，反弹可期
      S6. 深幅下跌后放量：N日跌幅>15% + 当日放量(>1.5)
                       → 恐慌抛售尾声，反转在即
      S7. 高位放量滞涨：价格处于高位 + 放量(>1.5) + 涨幅<1%
                       → 主力出货，警惕暴跌

      【量价异常 - 突破验证】
      S8. 缩量突破：突破N日新高 + 量不足(<1.0)
                  → 假突破，追高风险大
    """

    # ---- 可调参数 ----
    VOL_HIGH_THRESHOLD = 1.3    # 放量阈值（量比 > 此值 = 放量）
    VOL_LOW_THRESHOLD = 0.7     # 缩量阈值（量比 < 此值 = 缩量）
    VOL_MA_PERIOD = 20           # 均量计算周期
    PRICE_CHANGE_SMALL = 0.01    # 涨幅 < 1% = 滞涨
    DEEP_DROP_THRESHOLD = 0.15   # 深幅下跌阈值 15%
    DEEP_DROP_LOOKBACK = 30      # 深幅下跌回溯天数
    HIGH_POSITION_LOOKBACK = 60  # 高位判断回溯天数
    HIGH_POSITION_PERCENTILE = 0.85  # 价格处于回溯期前 85% = 高位
    DOUBLE_BOTTOM_LOOKBACK = 60  # 二次探底回溯天数
    DOUBLE_BOTTOM_VOL_RATIO = 0.6  # 二次探底量必须 < 前低量的 60%
    BREAKOUT_LOOKBACK = 60       # 突破新高回溯天数

    def __init__(self, **kwargs):
        """允许覆盖默认参数"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ============================================================
    #  辅助计算
    # ============================================================

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算衍生指标并返回新的 DataFrame"""
        data = df.copy()
        data['change_pct'] = data['close'].pct_change()
        data['volume_ma'] = data['volume'].rolling(self.VOL_MA_PERIOD).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['high_N'] = data['close'].rolling(self.BREAKOUT_LOOKBACK).max()
        data['low_N'] = data['close'].rolling(self.DOUBLE_BOTTOM_LOOKBACK).min()
        return data

    # ============================================================
    #  S1: 量价齐升
    # ============================================================
    def _check_s1(self, data: pd.DataFrame, i: int) -> Optional[Signal]:
        row = data.iloc[i]
        if row['change_pct'] > 0 and row['volume_ratio'] > self.VOL_HIGH_THRESHOLD:
            conf = min(100, row['volume_ratio'] * 40 + abs(row['change_pct']) * 300)
            return Signal(
                signal_id='S1',
                category='confirm',
                direction='bullish',
                confidence=round(conf, 1),
                description=f"量价齐升：涨幅 {row['change_pct']:.2%}，量比 {row['volume_ratio']:.1f}，资金积极进场",
                suggestion="上涨趋势健康，可持仓或顺势加仓，止损设于前日低点下方",
                date=str(row['date'].date()),
                price=row['close'],
                volume_ratio=row['volume_ratio']
            )
        return None

    # ============================================================
    #  S2: 缩量回调（价跌量缩 = 正常回调）
    # ============================================================
    def _check_s2(self, data: pd.DataFrame, i: int) -> Optional[Signal]:
        row = data.iloc[i]
        if row['change_pct'] < 0 and row['volume_ratio'] < self.VOL_LOW_THRESHOLD:
            return Signal(
                signal_id='S2',
                category='confirm',
                direction='neutral',
                confidence=round(min(80, (1 - row['volume_ratio']) * 80), 1),
                description=f"缩量回调：跌幅 {row['change_pct']:.2%}，量比 {row['volume_ratio']:.2f}，抛压枯竭",
                suggestion="回调缩量 = 正常调整，关注企稳信号，可在支撑位轻仓试多",
                date=str(row['date'].date()),
                price=row['close'],
                volume_ratio=row['volume_ratio']
            )
        return None

    # ============================================================
    #  S3: 价涨量缩 — 上涨乏力
    # ============================================================
    def _check_s3(self, data: pd.DataFrame, i: int) -> Optional[Signal]:
        row = data.iloc[i]
        if row['change_pct'] > 0 and row['volume_ratio'] < self.VOL_LOW_THRESHOLD:
            conf = min(90, (1 - row['volume_ratio']) * 80)
            return Signal(
                signal_id='S3',
                category='confirm',
                direction='bearish',
                confidence=round(conf, 1),
                description=f"价涨量缩：涨幅 {row['change_pct']:.2%}，量比 {row['volume_ratio']:.2f}，上涨动力不足",
                suggestion="无量上涨 = 虚涨，不宜追高。持多者应减仓或设紧止损",
                date=str(row['date'].date()),
                price=row['close'],
                volume_ratio=row['volume_ratio']
            )
        return None

    # ============================================================
    #  S4: 价跌量增 — 下跌加速
    # ============================================================
    def _check_s4(self, data: pd.DataFrame, i: int) -> Optional[Signal]:
        row = data.iloc[i]
        if row['change_pct'] < 0 and row['volume_ratio'] > self.VOL_HIGH_THRESHOLD:
            conf = min(95, row['volume_ratio'] * 35 + abs(row['change_pct']) * 200)
            return Signal(
                signal_id='S4',
                category='confirm',
                direction='bearish',
                confidence=round(conf, 1),
                description=f"价跌量增：跌幅 {row['change_pct']:.2%}，量比 {row['volume_ratio']:.1f}，恐慌抛售进行中",
                suggestion="放量下跌 = 下跌趋势强化，不宜抄底。空仓观望或严格止损",
                date=str(row['date'].date()),
                price=row['close'],
                volume_ratio=row['volume_ratio']
            )
        return None

    # ============================================================
    #  S5: 二次探底缩量 — 底部确认（关键反转信号）
    # ============================================================
    def _check_s5(self, data: pd.DataFrame, i: int) -> Optional[Signal]:
        row = data.iloc[i]
        lookback = self.DOUBLE_BOTTOM_LOOKBACK
        if i < lookback:
            return None

        # 找回溯期内最低点
        window = data.iloc[i - lookback:i]
        low_idx = window['close'].idxmin()
        low_row = data.loc[low_idx]

        # 条件1: 当前价格在前低的 3% 范围内
        near_low = abs(row['close'] - low_row['close']) / low_row['close'] < 0.03
        # 条件2: 当前成交量 < 前低成交量的 60%
        vol_shrink = row['volume'] < low_row['volume'] * self.DOUBLE_BOTTOM_VOL_RATIO
        # 条件3: 距离前低至少 10 天
        days_apart = i - low_idx > 10

        if near_low and vol_shrink and days_apart:
            conf = min(95, (1 - row['volume'] / low_row['volume']) * 100 + 30)
            return Signal(
                signal_id='S5',
                category='anomaly',
                direction='bullish',
                confidence=round(conf, 1),
                description=f"二次探底缩量：价格逼近前低 {low_row['close']:.2f}（{str(low_row['date'].date())}），"
                           f"量仅为前低的 {row['volume']/low_row['volume']:.0%}，抛压已尽",
                suggestion="经典底部信号！可在当前位置轻仓试多，止损设在今日最低点下方 2%",
                date=str(row['date'].date()),
                price=row['close'],
                volume_ratio=row['volume_ratio'],
                extra={'prev_low': low_row['close'], 'prev_low_date': str(low_row['date'].date()),
                       'vol_vs_prev': row['volume'] / low_row['volume']}
            )
        return None

    # ============================================================
    #  S6: 深幅下跌后放量 — 恐慌结束（反转信号）
    # ============================================================
    def _check_s6(self, data: pd.DataFrame, i: int) -> Optional[Signal]:
        row = data.iloc[i]
        lookback = self.DEEP_DROP_LOOKBACK
        if i < lookback:
            return None

        # 计算过去 N 日最高点到当前的跌幅
        window = data.iloc[i - lookback:i + 1]
        peak = window['close'].max()
        drop = (peak - row['close']) / peak

        if drop > self.DEEP_DROP_THRESHOLD and row['volume_ratio'] > 1.5:
            conf = min(95, drop * 300 + row['volume_ratio'] * 15)
            return Signal(
                signal_id='S6',
                category='anomaly',
                direction='bullish',
                confidence=round(conf, 1),
                description=f"深幅下跌后放量：{lookback}日内累计跌幅 {drop:.1%}，"
                           f"今日量比 {row['volume_ratio']:.1f}，恐慌盘涌出",
                suggestion="恐慌抛售尾声！放量 = 最后的多头投降，反转概率高。可分批建仓，止损放今日低点",
                date=str(row['date'].date()),
                price=row['close'],
                volume_ratio=row['volume_ratio'],
                extra={'total_drop': drop, 'peak_price': peak}
            )
        return None

    # ============================================================
    #  S7: 高位放量滞涨 — 主力出货（反转信号）
    # ============================================================
    def _check_s7(self, data: pd.DataFrame, i: int) -> Optional[Signal]:
        row = data.iloc[i]
        lookback = self.HIGH_POSITION_LOOKBACK
        if i < lookback:
            return None

        window = data.iloc[i - lookback:i + 1]
        # 处于回溯期高位（前 85%）
        percentile_rank = (window['close'] <= row['close']).mean()
        is_high = percentile_rank > self.HIGH_POSITION_PERCENTILE

        # 放量
        is_heavy_vol = row['volume_ratio'] > 1.5
        # 滞涨
        is_stalling = abs(row['change_pct']) < self.PRICE_CHANGE_SMALL

        if is_high and is_heavy_vol and is_stalling:
            conf = min(95, percentile_rank * 60 + (row['volume_ratio'] - 1) * 25)
            return Signal(
                signal_id='S7',
                category='anomaly',
                direction='bearish',
                confidence=round(conf, 1),
                description=f"高位放量滞涨：价格处于{lookback}日内前{int((1-percentile_rank)*100)}%高位，"
                           f"量比 {row['volume_ratio']:.1f} 但涨幅仅 {row['change_pct']:.2%}，主力在出货",
                suggestion="高位放量不涨 = 出货信号！持多者应立即减仓或清仓，不宜追高",
                date=str(row['date'].date()),
                price=row['close'],
                volume_ratio=row['volume_ratio'],
                extra={'percentile_rank': percentile_rank}
            )
        return None

    # ============================================================
    #  S8: 缩量突破 — 假突破（反转信号）
    # ============================================================
    def _check_s8(self, data: pd.DataFrame, i: int) -> Optional[Signal]:
        row = data.iloc[i]
        lookback = self.BREAKOUT_LOOKBACK
        if i < lookback:
            return None

        window = data.iloc[i - lookback:i]
        prev_high = window['close'].max()

        # 突破前高
        is_breakout = row['close'] > prev_high * 1.005  # 0.5% 缓冲
        # 量不足
        is_light_vol = row['volume_ratio'] < 1.0

        if is_breakout and is_light_vol:
            conf = min(85, (1 - row['volume_ratio']) * 60 + 30)
            prev_high_date = window.loc[window['close'].idxmax(), 'date']
            return Signal(
                signal_id='S8',
                category='anomaly',
                direction='bearish',
                confidence=round(conf, 1),
                description=f"缩量突破：价格突破{lookback}日新高（前高 {prev_high:.2f}），"
                           f"但量比仅 {row['volume_ratio']:.2f}",
                suggestion="无量突破 = 假突破概率大。不宜追涨，应等待放量确认后再入场",
                date=str(row['date'].date()),
                price=row['close'],
                volume_ratio=row['volume_ratio'],
                extra={'prev_high': prev_high, 'prev_high_date': str(prev_high_date.date())}
            )
        return None

    # ============================================================
    #  主扫描入口
    # ============================================================
    def scan(self, df: pd.DataFrame) -> List[Signal]:
        """
        扫描全部历史数据，返回所有触发的信号
        """
        data = self._prepare(df)
        signals = []
        for i in range(self.VOL_MA_PERIOD + self.BREAKOUT_LOOKBACK, len(data)):
            for checker in [self._check_s1, self._check_s2, self._check_s3,
                           self._check_s4, self._check_s5, self._check_s6,
                           self._check_s7, self._check_s8]:
                sig = checker(data, i)
                if sig:
                    signals.append(sig)
        # 按日期排序
        signals.sort(key=lambda s: s.date)
        return signals

    def scan_latest(self, df: pd.DataFrame) -> List[Signal]:
        """
        只扫描最后一天（用于实时监测）
        """
        data = self._prepare(df)
        signals = []
        i = len(data) - 1
        for checker in [self._check_s1, self._check_s2, self._check_s3,
                       self._check_s4, self._check_s5, self._check_s6,
                       self._check_s7, self._check_s8]:
            sig = checker(data, i)
            if sig:
                signals.append(sig)
        return signals

    def describe_all_signals(self) -> List[dict]:
        """返回所有 8 个信号的定义说明"""
        return [
            {
                'id': 'S1', 'name': '量价齐升', 'category': '量价确认',
                'direction': '看多', 'rule': '价涨 + 量比>1.3',
                'meaning': '上涨趋势健康，资金积极进场',
                'action': '持仓或顺势加仓'
            },
            {
                'id': 'S2', 'name': '缩量回调', 'category': '量价确认',
                'direction': '中性偏多', 'rule': '价跌 + 量比<0.7',
                'meaning': '抛压枯竭，回调接近尾声',
                'action': '等待企稳后试多'
            },
            {
                'id': 'S3', 'name': '价涨量缩', 'category': '量价确认',
                'direction': '看空', 'rule': '价涨 + 量比<0.7',
                'meaning': '无量上涨 = 虚涨，动力不足',
                'action': '不宜追高，持多者减仓'
            },
            {
                'id': 'S4', 'name': '价跌量增', 'category': '量价确认',
                'direction': '看空', 'rule': '价跌 + 量比>1.3',
                'meaning': '放量下跌，趋势加速恶化',
                'action': '不宜抄底，严格止损'
            },
            {
                'id': 'S5', 'name': '二次探底缩量', 'category': '量价异常',
                'direction': '看多★★★★★', 'rule': '价格≈前低 + 量<前低60%',
                'meaning': '底部确认，抛压已尽',
                'action': '经典底部信号，轻仓试多'
            },
            {
                'id': 'S6', 'name': '深跌后放量', 'category': '量价异常',
                'direction': '看多★★★★', 'rule': '30日跌>15% + 量比>1.5',
                'meaning': '恐慌抛售结束，最后一跌',
                'action': '分批建仓，止损放当日低点'
            },
            {
                'id': 'S7', 'name': '高位放量滞涨', 'category': '量价异常',
                'direction': '看空★★★★★', 'rule': '60日高位 + 量比>1.5 + 涨幅<1%',
                'meaning': '主力出货，量增价不增',
                'action': '立即减仓或清仓'
            },
            {
                'id': 'S8', 'name': '缩量突破', 'category': '量价异常',
                'direction': '看空★★★', 'rule': '突破60日新高 + 量比<1.0',
                'meaning': '无量突破 = 假突破',
                'action': '不追涨，等放量确认'
            }
        ]
