#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交易日 14:30 快照：上证指数实时点位/当日分时 + 全市场估值涨跌广度 + 覆盖基金估值。

输出 state/intraday.json，由 make_dashboard.py 读取生成“当日走势预测 + 价格分位 + 建议”。
只做一次东财请求，避免触发 push2 限流。
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, "state")
DST = os.path.join(STATE, "intraday.json")
BJ = timezone(timedelta(hours=8))

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Referer": "https://quote.eastmoney.com/",
}


def fetch_json(url, params, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            print(f"      请求失败({i + 1}/{tries})：{url.split('?')[0]} {exc}")
            time.sleep(1.2 * (i + 1))
    return None


INDEXES = [("1.000001", "上证指数"), ("1.000300", "沪深300")]


def index_snapshots() -> dict:
    """一次 ulist 批量取两个指数，返回 {'index': 上证, 'index2': 沪深300}。"""
    d = fetch_json("https://push2.eastmoney.com/api/qt/ulist.np/get",
                   {"secids": ",".join(s for s, _ in INDEXES),
                    "fields": "f2,f3,f12,f14,f15,f16,f18"})
    rows = ((d or {}).get("data") or {}).get("diff") or []
    out = {}
    keys = ["index", "index2"]
    for key, row in zip(keys, rows):
        try:
            price = float(row["f2"]) / 100
            pre = float(row["f18"]) / 100
            chg = float(row["f3"]) / 100
        except (TypeError, ValueError, KeyError):
            continue
        out[key] = {
            "name": row.get("f14") or dict(INDEXES).get("1." + str(row.get("f12")), ""),
            "price": round(price, 2), "preClose": round(pre, 2), "chg": round(chg, 2),
            "high": round(float(row["f15"]) / 100, 2) if row.get("f15") not in (None, "-") else None,
            "low": round(float(row["f16"]) / 100, 2) if row.get("f16") not in (None, "-") else None,
        }
    return out


def index_trend(secid: str) -> tuple[list, str]:
    """5 分钟 K 线近似当日分时；非交易时段返回最近一个交易日。"""
    d = fetch_json("https://push2his.eastmoney.com/api/qt/stock/kline/get",
                   {"secid": secid,
                    "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f53",
                    "klt": 5, "fqt": 0, "end": "20500101", "lmt": 120,
                    "ut": "fa5fd1943c7b386f172d6893dbfba10b"})
    klines = ((d or {}).get("data") or {}).get("klines") or []
    if not klines:
        return [], ""
    day = str(klines[-1]).split(" ")[0]
    out = []
    for row in klines:
        parts = str(row).split(",")
        if len(parts) < 2 or not parts[0].startswith(day):
            continue
        try:
            out.append([parts[0][-5:], round(float(parts[1]), 2)])
        except (TypeError, ValueError):
            continue
    return out, day


def index_from_kline(secid: str, name: str) -> dict | None:
    """兜底：用日线最后两根算点位与涨跌幅（盘中最后一根为实时值）。"""
    d = fetch_json("https://push2his.eastmoney.com/api/qt/stock/kline/get",
                   {"secid": secid,
                    "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f53",
                    "klt": 101, "fqt": 0, "end": "20500101", "lmt": 3,
                    "ut": "fa5fd1943c7b386f172d6893dbfba10b"})
    klines = ((d or {}).get("data") or {}).get("klines") or []
    closes = []
    for row in klines:
        parts = str(row).split(",")
        if len(parts) >= 2:
            try:
                closes.append(float(parts[1]))
            except ValueError:
                continue
    if len(closes) < 2 or not closes[-2]:
        return None
    price, pre = closes[-1], closes[-2]
    return {"name": name, "price": round(price, 2), "preClose": round(pre, 2),
            "chg": round((price / pre - 1) * 100, 2), "high": None, "low": None}


def valuation_snapshot(codes: set) -> tuple[dict, dict]:
    """东财全市场估算：返回 (覆盖基金估值涨跌, 广度统计)。"""
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        print(f"      akshare 不可用：{exc}")
        return {}, {}
    fn = getattr(ak, "fund_value_estimation_em", None)
    if fn is None:
        return {}, {}
    funds, up, down, flat, total = {}, 0, 0, 0, 0
    for symbol in ("全部",):
        try:
            df = fn(symbol=symbol)
        except Exception as exc:  # noqa: BLE001
            print(f"      全市场估值获取失败：{exc}")
            continue
        if df is None or df.empty:
            continue
        cols = {str(c): c for c in df.columns}
        c_code = next((cols[k] for k in cols if "基金代码" in k), None)
        c_chg = next((cols[k] for k in cols if "估算增长率" in k), None)
        if c_code is None or c_chg is None:
            continue
        for _, r in df.iterrows():
            raw = str(r[c_chg]).strip().replace("%", "")
            try:
                chg = float(raw)
            except (TypeError, ValueError):
                continue
            total += 1
            if chg > 0.01:
                up += 1
            elif chg < -0.01:
                down += 1
            else:
                flat += 1
            code = str(r[c_code]).zfill(6)
            if code in codes:
                funds[code] = round(chg, 2)
    breadth = {"up": up, "down": down, "flat": flat, "total": total,
               "upRatio": round(up / max(1, up + down) * 100, 1)}
    return funds, breadth


def listed_quotes(codes: list) -> dict:
    """场内 ETF/LOF 批量报价（ulist 一次可拿多只）。"""
    out = {}
    for i in range(0, len(codes), 50):
        batch = codes[i:i + 50]
        secids = ",".join(("1." if c[0] in "569" else "0.") + c for c in batch)
        d = fetch_json("https://push2.eastmoney.com/api/qt/ulist.np/get",
                       {"secids": secids, "fields": "f2,f3,f12,f14"})
        rows = ((d or {}).get("data") or {}).get("diff") or []
        for r in rows:
            try:
                chg = float(r["f3"]) / 100
            except (TypeError, ValueError, KeyError):
                continue
            out[str(r["f12"]).zfill(6)] = round(chg, 2)
        time.sleep(0.4)
    return out


def tier_codes() -> list:
    path = os.path.join(STATE, "tiers.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return []
    return [str(c).zfill(6) for c, v in data.items()
            if isinstance(v, dict) and v.get("tier") in ("S", "A", "B", "C")]


def main():
    ap = argparse.ArgumentParser(description="记录交易日盘中快照")
    ap.add_argument("--close", action="store_true", help="收盘修正：槽位记为 15:00")
    args = ap.parse_args()

    now = datetime.now(BJ)
    today = now.strftime("%Y-%m-%d")

    codes = tier_codes()
    print(f"      覆盖基金 {len(codes)} 只（S/A/B/C）")

    idxs = index_snapshots()
    for key, secid, name in (
            ("index", "1.000001", "上证指数"), ("index2", "1.000300", "沪深300")):
        if not idxs.get(key):
            print(f"      {name} 实时接口不可用，改用日线兜底")
            fb = index_from_kline(secid, name)
            if fb:
                idxs[key] = fb

    trends, days = {}, {}
    for key, secid, _ in (("index", "1.000001", "上证指数"),
                          ("index2", "1.000300", "沪深300")):
        trends[key], days[key] = index_trend(secid)
    trend_day = days.get("index") or days.get("index2") or ""
    if trend_day and trend_day != today:
        print(f"      分时日期 {trend_day} 非今日 {today}，疑似休市，不覆盖快照")
        return

    funds, breadth = valuation_snapshot(set(codes))
    listed = [c for c in codes if c.startswith("5") or c.startswith(("159", "15", "16", "18"))]
    if listed:
        quotes = listed_quotes(listed)
        funds.update(quotes)
        print(f"      场内报价 {len(quotes)} 只")

    payload = {
        "date": today,
        "time": now.strftime("%H:%M"),
        "stamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "index": dict(idxs.get("index") or {}, trend=trends["index"],
                      trendDate=days.get("index")),
        "index2": dict(idxs.get("index2") or {}, trend=trends["index2"],
                       trendDate=days.get("index2")),
        "breadth": breadth,
        "funds": funds,
    }

    old = {}
    if os.path.exists(DST):
        try:
            with open(DST, encoding="utf-8") as f:
                old = json.load(f)
        except Exception:  # noqa: BLE001
            old = {}
    slots = (old.get("slots") or []) if old.get("date") == today else []
    slots = slots[-5:]
    sse, csi = payload["index"], payload["index2"]
    if args.close:
        slots = [s for s in slots if not s.get("close")]
    slots.append({
        "time": "15:00" if args.close else payload["time"],
        "close": bool(args.close),
        "c1": sse.get("chg"), "i1": sse.get("price"),
        "c2": csi.get("chg"), "i2": csi.get("price"),
        "up": breadth.get("up"), "down": breadth.get("down"),
        "ratio": breadth.get("upRatio"),
    })
    payload["slots"] = slots

    os.makedirs(STATE, exist_ok=True)
    tmp = DST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, DST)
    print(f"已写入 {DST}：上证 {sse.get('price')}（{sse.get('chg')}%），"
          f"沪深300 {csi.get('price')}（{csi.get('chg')}%），"
          f"广度 {breadth.get('up')}/{breadth.get('down')}，基金 {len(funds)} 只")


if __name__ == "__main__":
    main()
