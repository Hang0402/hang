# -*- coding: utf-8 -*-
"""
回测系统 - 基于量价信号的交易模拟
评估策略表现：胜率、盈亏比、最大回撤、夏普比率
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .signal_engine import VolumePriceEngine, Signal


@dataclass
class Trade:
    """单笔交易记录"""
    entry_date: str
    exit_date: str
    direction: str       # 'long' / 'short'
    entry_price: float
    exit_price: float
    pnl_pct: float       # 盈亏百分比
    signal_id: str       # 触发信号
    exit_reason: str     # 出场原因


@dataclass
class BacktestResult:
    """回测结果"""
    symbol: str
    start_date: str
    end_date: str
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float           # 胜率
    avg_win_pct: float        # 平均盈利
    avg_loss_pct: float       # 平均亏损
    profit_factor: float      # 盈亏比
    total_return: float       # 总收益
    max_drawdown: float       # 最大回撤
    sharpe_ratio: float       # 夏普比率（简化版）
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class VolumePriceBacktest:
    """
    量价策略回测器

    交易规则：
      - 看多信号(S1/S5/S6): 次日开盘买入，持有 N 天或触发止损/止盈
      - 看空信号(S3/S4/S7/S8): 次日开盘卖出（如有持仓），或做空标记
      - 止损: -5%
      - 止盈: +15%
      - 最大持仓天数: 20 天
    """

    def __init__(self,
                 stop_loss: float = -0.05,
                 take_profit: float = 0.15,
                 max_hold_days: int = 20,
                 capital: float = 100000):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_days = max_hold_days
        self.capital = capital
        self.engine = VolumePriceEngine()

    def run(self, df: pd.DataFrame, symbol: str = '') -> BacktestResult:
        """
        执行回测
        """
        signals = self.engine.scan(df)
        trades = []
        equity = [self.capital]
        position = 0          # 持仓数量（0 = 空仓）
        entry_price = 0.0
        entry_date = ''
        entry_signal = ''
        hold_days = 0

        # 建立日期索引，方便查找
        date_to_idx = {str(d.date()): i for i, d in enumerate(df['date'])}

        for sig in signals:
            # 跳过无对应日期的信号
            if sig.date not in date_to_idx:
                continue
            sig_idx = date_to_idx[sig.date]

            if position > 0:
                # 已有持仓：检查是否出场
                hold_days += 1
                exit_idx = sig_idx
                if exit_idx >= len(df):
                    continue
                current_price = df.iloc[exit_idx]['close']
                pnl = (current_price - entry_price) / entry_price

                exit_reason = ''
                should_exit = False

                # 检查止损
                if pnl <= self.stop_loss:
                    exit_reason = '止损'
                    should_exit = True
                # 检查止盈
                elif pnl >= self.take_profit:
                    exit_reason = '止盈'
                    should_exit = True
                # 检查最大持仓天数
                elif hold_days >= self.max_hold_days:
                    exit_reason = '到期'
                    should_exit = True
                # 看空信号出现：平多仓
                elif sig.direction == 'bearish' and sig.confidence > 60:
                    exit_reason = f'看空信号({sig.signal_id})'
                    should_exit = True
                # S3 价涨量缩：减仓信号
                elif sig.signal_id == 'S3' and sig.confidence > 50:
                    exit_reason = f'上涨乏力({sig.signal_id})'
                    should_exit = True

                if should_exit:
                    trade = Trade(
                        entry_date=entry_date,
                        exit_date=sig.date,
                        direction='long',
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl_pct=round(pnl * 100, 2),
                        signal_id=entry_signal,
                        exit_reason=exit_reason
                    )
                    trades.append(trade)
                    self.capital *= (1 + pnl)
                    position = 0
                    hold_days = 0

            # 开仓条件：看多信号 + 无持仓
            if position == 0 and sig.direction == 'bullish' and sig.confidence > 50:
                # 次日开盘买入
                next_idx = sig_idx + 1
                if next_idx >= len(df):
                    continue
                entry_price = df.iloc[next_idx]['open']
                entry_date = str(df.iloc[next_idx]['date'].date())
                entry_signal = sig.signal_id
                position = 1
                hold_days = 0

            # 记录权益曲线
            if position > 0:
                current_val = self.capital * (1 + (df.iloc[sig_idx]['close'] - entry_price) / entry_price)
                equity.append(current_val)
            else:
                equity.append(self.capital)

        # 强制平仓
        if position > 0:
            last_price = df.iloc[-1]['close']
            pnl = (last_price - entry_price) / entry_price
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=str(df.iloc[-1]['date'].date()),
                direction='long',
                entry_price=entry_price,
                exit_price=last_price,
                pnl_pct=round(pnl * 100, 2),
                signal_id=entry_signal,
                exit_reason='回测结束'
            ))
            self.capital *= (1 + pnl)

        # ---- 统计 ----
        total = len(trades)
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total * 100 if total > 0 else 0
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
        profit_factor = abs(avg_win * win_count / (avg_loss * loss_count)) if loss_count and avg_loss else 999

        total_return = (self.capital / 100000 - 1) * 100

        # 最大回撤
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak
        max_dd = drawdown.min() * 100

        # 简化的夏普比率
        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return BacktestResult(
            symbol=symbol,
            start_date=str(df.iloc[0]['date'].date()),
            end_date=str(df.iloc[-1]['date'].date()),
            total_trades=total,
            win_trades=win_count,
            loss_trades=loss_count,
            win_rate=round(win_rate, 2),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            total_return=round(total_return, 2),
            max_drawdown=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            trades=trades,
            equity_curve=[round(e, 2) for e in equity]
        )
