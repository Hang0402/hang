# -*- coding: utf-8 -*-
"""四层筛选 + 综合评分模块 —— quantipy 集成版"""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

# 从 quantipy 配置（如有），否则用默认值
try:
    from config_screen import (
        FILTER_ROE_MIN, FILTER_DEBT_MAX,
        FILTER_GAIN_20D_MIN, FILTER_TURNOVER_MIN,
        WEIGHT_TREND, WEIGHT_VOLUME, WEIGHT_FINANCIAL, WEIGHT_VALUATION,
        TOP_N,
    )
except ImportError:
    FILTER_ROE_MIN = 10
    FILTER_DEBT_MAX = 60
    FILTER_GAIN_20D_MIN = 10
    FILTER_TURNOVER_MIN = 3
    WEIGHT_TREND = 0.40
    WEIGHT_VOLUME = 0.25
    WEIGHT_FINANCIAL = 0.20
    WEIGHT_VALUATION = 0.15
    TOP_N = 10

console = Console()


def check_trend(daily_df):
    """检查趋势：价格 > MA20 > MA60，20日涨幅 > 10%"""
    if daily_df is None or len(daily_df) < 60:
        return False, {}

    latest = daily_df.iloc[-1]
    close = latest["close"]
    ma20 = latest.get("MA20", np.nan)
    ma60 = latest.get("MA60", np.nan)
    gain_20d = latest.get("gain_20d", np.nan)

    if pd.isna(ma20) or pd.isna(ma60) or pd.isna(gain_20d):
        return False, {}

    ok = (close > ma20) and (ma20 > ma60) and (gain_20d > FILTER_GAIN_20D_MIN)
    trend_score = min(gain_20d / 30 * 100, 100) if gain_20d > 0 else 0

    return ok, {
        "trend_score": trend_score, "close": close,
        "ma20": ma20, "ma60": ma60, "gain_20d": gain_20d,
    }


def check_capital(daily_df):
    """检查资金：换手率 > 3%，成交量放大"""
    if daily_df is None or len(daily_df) < 20:
        return False, {}

    latest = daily_df.iloc[-1]
    turnover = latest.get("turnover", np.nan)
    vol_ma5 = latest.get("vol_ma5", np.nan)
    vol_ma20 = latest.get("vol_ma20", np.nan)

    if pd.isna(turnover) or pd.isna(vol_ma5) or pd.isna(vol_ma20):
        return False, {}

    vol_expand = vol_ma5 > vol_ma20
    ok = (turnover > FILTER_TURNOVER_MIN) and vol_expand

    turnover_score = min(turnover / 15 * 100, 100)
    vol_ratio = (vol_ma5 / vol_ma20 - 1) * 100 if vol_ma20 > 0 else 0
    vol_expand_score = min(max(vol_ratio * 5, 0), 100)
    volume_score = (turnover_score + vol_expand_score) / 2

    return ok, {
        "volume_score": volume_score, "turnover": turnover,
        "vol_ratio": vol_ratio,
    }


def composite_score(financial_data, trend_info, capital_info, price=None):
    """综合评分：趋势40% + 成交量25% + 财务20% + 估值15%"""
    trend_score = trend_info.get("trend_score", 0)
    volume_score = capital_info.get("volume_score", 0)

    fin = financial_data or {}
    roe = fin.get("ROE") or 0
    debt = fin.get("debt_ratio") or 100

    roe_score = min(roe / 30 * 100, 100)
    debt_score = max((1 - debt / 80) * 100, 0) if debt > 0 else 50
    financial_score = (roe_score * 0.6 + debt_score * 0.4)

    eps = fin.get("EPS")
    bvps = fin.get("BVPS")
    close_price = price if price else trend_info.get("close", 0)

    pe = None; pb = None
    if eps and eps > 0 and close_price > 0:
        pe = close_price / eps
    if bvps and bvps > 0 and close_price > 0:
        pb = close_price / bvps

    pe_score = min(30 / pe * 100, 100) if (pe and pe > 0) else 50
    pb_score = min(3 / pb * 100, 100) if (pb and pb > 0) else 50
    valuation_score = (pe_score * 0.5 + pb_score * 0.5)

    total = (trend_score * WEIGHT_TREND + volume_score * WEIGHT_VOLUME +
             financial_score * WEIGHT_FINANCIAL + valuation_score * WEIGHT_VALUATION)

    return {
        "total": round(total, 1),
        "trend_score": round(trend_score, 1),
        "volume_score": round(volume_score, 1),
        "financial_score": round(financial_score, 1),
        "valuation_score": round(valuation_score, 1),
        "roe": roe, "debt_ratio": debt, "pe": pe, "pb": pb,
    }


def print_results(results):
    """打印 Top N"""
    if not results:
        console.print("[red]没有股票通过全部筛选条件[/red]")
        return []

    sorted_results = sorted(results, key=lambda x: x["score"]["total"], reverse=True)
    top_n = sorted_results[:TOP_N]

    table = Table(title=f"Top {TOP_N} 综合评分排名")
    table.add_column("#", style="bold cyan", width=4, justify="center")
    table.add_column("代码", style="bold white", width=8)
    table.add_column("名称", style="bold green", width=10)
    table.add_column("总分", style="bold yellow", width=6, justify="center")
    table.add_column("趋势", width=5)
    table.add_column("量能", width=5)
    table.add_column("财务", width=5)
    table.add_column("估值", width=5)
    table.add_column("ROE%", width=6)
    table.add_column("PE", width=7)
    table.add_column("20日涨%", width=7)
    table.add_column("换手%", width=6)

    for i, r in enumerate(top_n, 1):
        s = r["score"]
        table.add_row(
            str(i), r["code"], r["name"],
            f"[bold]{s['total']}[/bold]",
            f"{s['trend_score']:.0f}", f"{s['volume_score']:.0f}",
            f"{s['financial_score']:.0f}", f"{s['valuation_score']:.0f}",
            f"{s['roe']:.1f}" if s["roe"] else "-",
            f"{s['pe']:.1f}" if s["pe"] else "-",
            f"{r['trend']['gain_20d']:.1f}" if r["trend"]["gain_20d"] else "-",
            f"{r['capital']['turnover']:.1f}" if r["capital"]["turnover"] else "-",
        )

    console.print(table)
    console.print(f"\n[dim]共 {len(sorted_results)} 支通过全部条件，展示前 {TOP_N} 支[/dim]")
    return top_n


def to_dict_list(results):
    """将结果转为可 JSON 序列化的字典列表"""
    out = []
    sorted_results = sorted(results, key=lambda x: x["score"]["total"], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        s = r["score"]
        out.append({
            "rank": i,
            "code": r["code"],
            "name": r["name"],
            "industry": r.get("industry", ""),
            "total": s["total"],
            "trend_score": s["trend_score"],
            "volume_score": s["volume_score"],
            "financial_score": s["financial_score"],
            "valuation_score": s["valuation_score"],
            "roe": round(s["roe"], 1) if s["roe"] else None,
            "debt_ratio": round(s["debt_ratio"], 1) if s["debt_ratio"] else None,
            "pe": round(s["pe"], 1) if s["pe"] else None,
            "pb": round(s["pb"], 1) if s["pb"] else None,
            "gain_20d": round(r["trend"]["gain_20d"], 1) if r["trend"]["gain_20d"] else None,
            "turnover": round(r["capital"]["turnover"], 1) if r["capital"]["turnover"] else None,
            "ma20": round(r["trend"]["ma20"], 2),
            "ma60": round(r["trend"]["ma60"], 2),
            "close": round(r["trend"]["close"], 2),
            "vol_ratio": round(r["capital"]["vol_ratio"], 1),
        })
    return out
