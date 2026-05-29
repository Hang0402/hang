# -*- coding: utf-8 -*-
"""数据获取模块 - akshare + 模拟数据回退"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

try:
    import akshare as ak
except ImportError:
    ak = None


def _code_to_ak_symbol(code: str) -> str:
    code = code.strip().zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    elif code.startswith(("0", "3")):
        return f"sz{code}"
    elif code.startswith(("4", "8")):
        return f"bj{code}"
    return f"sz{code}"


def _gen_demo(code: str, start: str, end: str) -> pd.DataFrame:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    dates = pd.date_range(start=start, end=end, freq="B")
    if len(dates) < 10:
        dates = pd.date_range(start=start, periods=60, freq="B")
    n = len(dates)
    np.random.seed(hash(code) % 2**31)
    trend = np.linspace(0, np.random.uniform(-0.3, 0.5), n)
    noise = np.random.randn(n) * 0.02
    returns = np.concatenate([[0], np.diff(trend)]) * 0.3 + noise
    close = 20 + np.exp(np.cumsum(returns)) * 15
    base_vol = np.random.randint(5000000, 20000000, n).astype(float)
    vol_spike = np.random.rand(n) < 0.12
    base_vol[vol_spike] *= np.random.uniform(1.5, 3.0, vol_spike.sum())
    high = close * (1 + np.abs(np.random.randn(n) * 0.015))
    low = close * (1 - np.abs(np.random.randn(n) * 0.015))
    open_p = low + np.random.rand(n) * (high - low)
    df = pd.DataFrame({
        "date": dates, "open": open_p, "high": high,
        "low": low, "close": close, "volume": base_vol,
        "amount": base_vol * close
    })
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_daily(code: str, start: str = "2020-01-01",
                end: Optional[str] = None) -> pd.DataFrame:
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # Always try demo first if ak unavailable
    if ak is None:
        return _gen_demo(code, start, end)

    try:
        symbol = _code_to_ak_symbol(code)
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq"
        )
        if df is None or df.empty:
            raise ValueError("empty data")
    except Exception:
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq"
            )
            if df is None or df.empty:
                raise ValueError("empty data")
        except Exception:
            return _gen_demo(code, start, end)

    # Normalize columns
    col_map = {
        "??": "date", "??": "open", "??": "high",
        "??": "low", "??": "close", "???": "volume",
        "???": "amount"
    }
    df.rename(columns=col_map, inplace=True)

    # Ensure required columns exist, fall back to demo if not
    needed = ["date", "open", "high", "low", "close", "volume", "amount"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return _gen_demo(code, start, end)

    df = df[needed].copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df
def fetch_realtime(code: str) -> dict:
    if ak is None:
        return {
            "code": code, "name": code, "price": 0, "change_pct": 0,
            "volume": 0, "amount": 0, "high": 0, "low": 0, "open": 0,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    try:
        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == code.strip().zfill(6)]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {
            "code": code,
            "name": r.get("名称", ""),
            "price": float(r["最新价"]),
            "change_pct": float(r["涨跌幅"]),
            "volume": float(r["成交量"]),
            "amount": float(r["成交额"]),
            "high": float(r["最高"]),
            "low": float(r["最低"]),
            "open": float(r["今开"]),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception:
        return {}
