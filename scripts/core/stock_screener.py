# -*- coding: utf-8 -*-
"""股票筛选数据获取 —— quantipy 集成版
基于 Sina 日线 + THS 财务 + 交易所列表 + 新浪直连快照"""

import time
import re
import akshare as ak
import pandas as pd
import numpy as np
import requests

REQUEST_DELAY = 0.3
MAX_RETRIES = 2


def _retry(fn, name, *args, **kwargs):
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = fn(*args, **kwargs)
            time.sleep(REQUEST_DELAY)
            return result
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1)
    return None


def get_stock_list():
    """获取全部A股，自动识别ST并排除"""
    sz = _retry(ak.stock_info_sz_name_code, "深交所")
    sh = _retry(ak.stock_info_sh_name_code, "上交所")

    frames = []
    if sz is not None:
        sz = sz.rename(columns={"A股代码": "code", "A股简称": "name", "所属行业": "industry"})
        sz["code"] = sz["code"].astype(str).str.zfill(6)
        frames.append(sz[["code", "name", "industry"]])
    if sh is not None:
        sh = sh.rename(columns={"证券代码": "code", "证券简称": "name"})
        sh["code"] = sh["code"].astype(str).str.zfill(6)
        sh["industry"] = ""
        frames.append(sh[["code", "name", "industry"]])

    if not frames:
        return pd.DataFrame()

    all_stocks = pd.concat(frames, ignore_index=True)
    all_stocks["is_st"] = all_stocks["name"].str.contains(r"\bST\b|\*ST", na=False)
    clean = all_stocks[~all_stocks["is_st"]].copy()
    return clean


def get_spot_snapshot(codes):
    """新浪直连实时行情，按成交额排序"""
    SINA_URL = "https://hq.sinajs.cn/list="
    headers = {"Referer": "https://finance.sina.com.cn"}
    BATCH = 800
    results = []

    for start in range(0, len(codes), BATCH):
        batch = codes[start:start + BATCH]
        symbols = []
        for c in batch:
            if c.startswith(("0", "3")):
                symbols.append(f"sz{c}")
            elif c.startswith("6"):
                symbols.append(f"sh{c}")
            else:
                symbols.append(f"sz{c}")

        try:
            r = requests.get(SINA_URL + ",".join(symbols), headers=headers, timeout=30)
            r.encoding = "gbk"
            for line in r.text.strip().split("\n"):
                m = re.search(r'hq_str_(\w+)="(.+)"', line)
                if not m:
                    continue
                data = m.group(2).split(",")
                if len(data) < 9:
                    continue
                try:
                    price = float(data[3]) if data[3] else 0
                    amount = float(data[8]) * 10000 if data[8] else 0
                    if amount == 0:
                        continue
                    prev_close = float(data[2]) if data[2] else 0
                    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
                    results.append({
                        "code": m.group(1)[2:], "name": data[0],
                        "price": price, "amount": amount,
                        "amount_num": amount, "change_pct": change_pct,
                    })
                except (ValueError, IndexError):
                    continue
            time.sleep(0.2)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.dropna(subset=["amount_num"])
    return df.sort_values("amount_num", ascending=False)


def get_stock_daily(code):
    """获取个股日线（新浪源），含 MA/涨跌幅/量比，turnover 已转百分比"""
    if code.startswith(("0", "3")):
        symbol = f"sz{code}"
    elif code.startswith("6"):
        symbol = f"sh{code}"
    else:
        return None

    try:
        df = ak.stock_zh_a_daily(symbol=symbol, start_date="20250101",
                                  end_date="20500101", adjust="qfq")
        if df is None or df.empty:
            return None
        df = df.sort_values("date").reset_index(drop=True)
        df["turnover"] = df["turnover"] * 100
        df["MA20"] = df["close"].rolling(20).mean()
        df["MA60"] = df["close"].rolling(60).mean()
        if len(df) >= 21:
            df["gain_20d"] = (df["close"] / df["close"].shift(20) - 1) * 100
        else:
            df["gain_20d"] = None
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        return df
    except Exception:
        return None


def get_financial_data(code):
    """获取财务数据：ROE/负债率/EPS/每股净资产，优先取最新年报"""
    try:
        df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        if df is None or df.empty:
            return None

        def _p(v):
            if v is None or str(v).lower() in ("false", "nan", ""):
                return None
            s = str(v).replace("%", "").replace(",", "").strip()
            try: return float(s)
            except: return None

        best = None
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            roe_v = row.get("净资产收益率", None)
            if str(roe_v).lower() in ("false", "nan", ""):
                continue
            if "-12-31" in str(row["报告期"]):
                best = row; break
        if best is None:
            for i in range(len(df) - 1, -1, -1):
                row = df.iloc[i]
                if str(row.get("净资产收益率", "")).lower() not in ("false", "nan", ""):
                    best = row; break

        if best is None:
            return None
        return {
            "ROE": _p(best.get("净资产收益率")),
            "debt_ratio": _p(best.get("资产负债率")),
            "EPS": _p(best.get("基本每股收益")),
            "BVPS": _p(best.get("每股净资产")),
        }
    except Exception:
        return None
