# -*- coding: utf-8 -*-
"""Flask API - 量价关系量化系统 + 四层股票筛选"""

import sys, os, traceback, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from datetime import datetime, timedelta

from core.data_fetcher import fetch_daily, fetch_realtime
from core.signal_engine import VolumePriceEngine
from core.backtest import VolumePriceBacktest
from core.screener import check_trend, check_capital, composite_score, to_dict_list
from core.stock_screener import (
    get_stock_list, get_spot_snapshot, get_stock_daily as screen_daily,
    get_financial_data as screen_financial,
)

app = Flask(__name__)
CORS(app)
engine = VolumePriceEngine()
backtester = VolumePriceBacktest()

# 筛选阈值
FILTER_ROE_MIN = 10
FILTER_DEBT_MAX = 60

# 缓存股票列表
_stock_cache = {"list": None, "code_to_name": {}, "codes": []}


def _load_stock_cache():
    if _stock_cache["list"] is not None:
        return
    sl = get_stock_list()
    if sl is not None and not sl.empty:
        _stock_cache["list"] = sl
        _stock_cache["code_to_name"] = dict(zip(sl["code"], sl["name"]))
        _stock_cache["codes"] = sl["code"].tolist()


def clean(obj):
    if isinstance(obj, float):
        return 0.0 if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def ok(data):
    return jsonify(clean(data))


# ─── 原有 API ─────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/signal_definitions")
def signal_definitions():
    return ok({"signals": engine.describe_all_signals()})


@app.route("/api/scan")
def scan():
    code = request.args.get("code", "600519")
    start = request.args.get("start", "2024-01-01")
    end = request.args.get("end", datetime.today().strftime("%Y-%m-%d"))
    try:
        df = fetch_daily(code, start, end)
        signals = engine.scan(df)
        return ok({
            "code": code, "data_points": len(df),
            "total_signals": len(signals),
            "bullish": sum(1 for s in signals if s.direction == "bullish"),
            "bearish": sum(1 for s in signals if s.direction == "bearish"),
            "neutral": sum(1 for s in signals if s.direction == "neutral"),
            "signals": [{
                "signal_id": s.signal_id, "category": s.category,
                "direction": s.direction, "confidence": s.confidence,
                "description": s.description, "suggestion": s.suggestion,
                "date": s.date, "price": s.price,
                "volume_ratio": round(s.volume_ratio, 2), "extra": s.extra
            } for s in signals[-50:]]
        })
    except Exception as e:
        traceback.print_exc()
        return ok({"error": str(e)}), 500


@app.route("/api/monitor")
def monitor():
    code = request.args.get("code", "600519")
    start = (datetime.today() - timedelta(days=120)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    try:
        df = fetch_daily(code, start, end)
        signals = engine.scan_latest(df)
        realtime = fetch_realtime(code)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change_pct = (latest["close"] - prev["close"]) / prev["close"] * 100
        vol_ma = df["volume"].rolling(20).mean().iloc[-1]
        vol_ratio = latest["volume"] / vol_ma if vol_ma and vol_ma > 0 else 1.0
        return ok({
            "code": code, "name": realtime.get("name", ""),
            "price": round(float(latest["close"]), 2),
            "change_pct": round(float(change_pct), 2),
            "volume_ratio": round(float(vol_ratio), 2),
            "date": str(latest["date"].date()),
            "signals": [{
                "signal_id": s.signal_id, "category": s.category,
                "direction": s.direction, "confidence": s.confidence,
                "description": s.description, "suggestion": s.suggestion,
                "extra": s.extra
            } for s in signals],
            "signal_count": len(signals),
            "has_alert": any(s.confidence > 60 for s in signals)
        })
    except Exception as e:
        traceback.print_exc()
        return ok({"error": str(e)}), 500


@app.route("/api/backtest")
def backtest():
    code = request.args.get("code", "600519")
    start = request.args.get("start", "2023-01-01")
    end = request.args.get("end", datetime.today().strftime("%Y-%m-%d"))
    try:
        df = fetch_daily(code, start, end)
        result = backtester.run(df, symbol=code)
        return ok({
            "symbol": result.symbol, "start_date": result.start_date,
            "end_date": result.end_date, "total_trades": result.total_trades,
            "win_trades": result.win_trades, "loss_trades": result.loss_trades,
            "win_rate": result.win_rate, "avg_win_pct": result.avg_win_pct,
            "avg_loss_pct": result.avg_loss_pct,
            "profit_factor": result.profit_factor,
            "total_return": result.total_return,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "trades": [{
                "entry_date": t.entry_date, "exit_date": t.exit_date,
                "direction": t.direction, "entry_price": t.entry_price,
                "exit_price": t.exit_price, "pnl_pct": t.pnl_pct,
                "signal_id": t.signal_id, "exit_reason": t.exit_reason
            } for t in result.trades[-30:]],
            "equity_curve": result.equity_curve[::max(1, len(result.equity_curve)//200)]
        })
    except Exception as e:
        traceback.print_exc()
        return ok({"error": str(e)}), 500


@app.route("/api/history")
def history():
    code = request.args.get("code", "600519")
    start = request.args.get("start", "2024-01-01")
    end = request.args.get("end", datetime.today().strftime("%Y-%m-%d"))
    try:
        df = fetch_daily(code, start, end)
        vol_ma = df["volume"].rolling(20).mean()
        vol_ratio = df["volume"] / vol_ma.replace(0, 1)
        return ok({
            "code": code,
            "data": [{
                "date": str(r["date"].date()),
                "open": round(float(r["open"]), 2),
                "high": round(float(r["high"]), 2),
                "low": round(float(r["low"]), 2),
                "close": round(float(r["close"]), 2),
                "volume": int(r["volume"]),
                "volume_ratio": round(float(vol_ratio.iloc[i]), 2) if not math.isnan(vol_ratio.iloc[i]) and not math.isinf(vol_ratio.iloc[i]) else 1.0
            } for i, (_, r) in enumerate(df.iterrows())]
        })
    except Exception as e:
        traceback.print_exc()
        return ok({"error": str(e)}), 500


# ─── 新增：四层筛选 API ──────────────────────

@app.route("/api/screen/run")
def screen_run():
    """执行四层筛选"""
    try:
        _load_stock_cache()
        codes = _stock_cache["codes"]
        c2n = _stock_cache["code_to_name"]

        spot = get_spot_snapshot(codes)
        if spot.empty:
            return ok({"error": "获取实时行情失败"}), 500

        # 取前 200 活跃股
        spot = spot[spot["code"].isin(codes)]
        candidates = spot.head(200)["code"].tolist()

        results = []
        stats = {"l1_pass": 0, "l2_pass": 0, "l3_pass": 0,
                 "l1_fail": 0, "l2_fail": 0, "l3_fail": 0, "fetch_fail": 0}

        for code in candidates:
            name = c2n.get(code, "?")
            fin = screen_financial(code)
            if fin is None:
                stats["fetch_fail"] += 1
                continue
            roe = fin.get("ROE"); debt = fin.get("debt_ratio")
            if roe is None or debt is None:
                stats["fetch_fail"] += 1
                continue
            if roe < FILTER_ROE_MIN or debt > FILTER_DEBT_MAX:
                stats["l1_fail"] += 1
                continue
            stats["l1_pass"] += 1

            daily = screen_daily(code)
            if daily is None:
                stats["fetch_fail"] += 1; stats["l1_pass"] -= 1
                continue

            t_ok, t_info = check_trend(daily)
            if not t_ok:
                stats["l2_fail"] += 1
                continue
            stats["l2_pass"] += 1

            c_ok, c_info = check_capital(daily)
            if not c_ok:
                stats["l3_fail"] += 1
                continue
            stats["l3_pass"] += 1

            score = composite_score(fin, t_info, c_info, price=t_info.get("close"))
            results.append({
                "code": code, "name": name,
                "trend": t_info, "capital": c_info,
                "financial": fin, "score": score,
                "industry": "",
            })

        return ok({
            "status": "ok",
            "total_candidates": len(candidates),
            "final_count": len(results),
            "stats": stats,
            "stocks": to_dict_list(results),
        })
    except Exception as e:
        traceback.print_exc()
        return ok({"error": str(e)}), 500


@app.route("/api/screen/stock/<code>")
def screen_stock_detail(code):
    """获取单支股票筛选详情"""
    try:
        _load_stock_cache()
        name = _stock_cache["code_to_name"].get(code, code)

        daily = screen_daily(code)
        fin = screen_financial(code)
        if daily is None:
            return ok({"error": "无行情数据"}), 404

        t_ok, t_info = check_trend(daily)
        c_ok, c_info = check_capital(daily)
        score = composite_score(fin, t_info, c_info, price=t_info.get("close"))

        # 近5日走势
        recent = daily.tail(5)
        recent_data = []
        base_price = recent.iloc[0]["close"]
        for _, r in recent.iterrows():
            chg = ((r["close"] - base_price) / base_price * 100) if base_price else 0
            recent_data.append({
                "date": str(r["date"].date())[-5:],
                "close": round(float(r["close"]), 2),
                "change_pct": round(float(chg), 2),
                "turnover": round(float(r.get("turnover", 0)), 2),
                "volume": int(r["volume"]),
            })

        return ok({
            "code": code, "name": name,
            "trend_ok": t_ok, "capital_ok": c_ok,
            "trend": {
                "close": round(t_info.get("close", 0), 2),
                "ma20": round(t_info.get("ma20", 0), 2),
                "ma60": round(t_info.get("ma60", 0), 2),
                "gain_20d": round(t_info.get("gain_20d", 0) or 0, 1),
            },
            "capital": {
                "turnover": round(c_info.get("turnover", 0), 2),
                "vol_ratio": round(c_info.get("vol_ratio", 0), 1),
            },
            "score": clean(score),
            "recent_5d": recent_data,
        })
    except Exception as e:
        traceback.print_exc()
        return ok({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  量价关系量化交易系统")
    print("  地址: http://localhost:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
