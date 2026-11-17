#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成基金看板：左侧小卡片 + 右侧常驻详情栏。

新增数据（相较旧版）
  1. 近 1 年 / 近 2 年：收益率、年化波动、夏普（用 cache/nav_store.pkl 的累计净值现算）
  2. 同类分位：同 fund_type 分组内的夏普、1 年收益、2 年收益、1 年波动分位（稳定性越低越好）
  3. 区间高低：近 1 年 / 近 3 年最高最低净值、距今多少天、当前距离高低点百分比
  4. 行业配置：东财 F10 行业配置（仅 S/A/B/C，落盘 cache/industry.json 增量缓存）
  5. 场外估值：东财全市场估算净值快照（构建时一次，页面标注非实时）
场内 ETF/LOF 的实时价与当日分时由前端 JSONP 直接请求东财 push2，不在此处抓取。
"""
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
CACHE = os.path.join(BASE, "cache")
TIERS = ["S", "A", "B", "C", "D", "E"]
TIER_DESC = {
    "S": "头部优质：高收益、低回撤、风格稳定",
    "A": "潜力增长：风险调整后收益良好",
    "B": "表现平庸：收益一般，波动偏大",
    "C": "不建议：高风险低收益",
    "D": "未过筛：未通过第一道门",
    "E": "退出：规模过小或数据不足",
}
DIM_KEYS = ["ret_power", "dd_power", "calmar_power", "stab_power",
            "style_power", "mgr_power", "scale_power"]
DIM_NAMES = ["收益", "抗跌", "性价比", "稳定", "风格", "经理", "规模"]
RF = 0.02
INDUSTRY_CACHE = os.path.join(CACHE, "industry.json")
INTRADAY = os.path.join(BASE, "state", "intraday.json")


def read(name, **kw):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"fund_code": str}, **kw)


def num(v, nd=3):
    try:
        f = float(v)
        return None if pd.isna(f) else round(f, nd)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 净值窗口

def load_nav_store() -> dict:
    path = os.path.join(CACHE, "nav_store.pkl")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"      净值缓存读取失败：{exc}")
        return {}


def window_stats(s: pd.Series, days: int) -> list | None:
    """近 N 个交易日：收益 / 年化波动 / 夏普。"""
    seg = s.iloc[-days:]
    if len(seg) < 60:
        return None
    ret = float(seg.iloc[-1] / seg.iloc[0] - 1)
    vol = float(seg.pct_change().std() * np.sqrt(252))
    if not np.isfinite(vol) or vol <= 0:
        return None
    sharpe = (ret - RF * days / 252) / (vol * np.sqrt(days / 252))
    return [round(ret, 4), round(vol, 4), round(float(sharpe), 2)]


def price_range(s: pd.Series, days: int) -> list | None:
    """近 N 日：最高、最高距今天数、最低、最低距今天数、当前距高%、当前距低%。"""
    seg = s.iloc[-days:]
    if len(seg) < 60:
        return None
    cur = float(seg.iloc[-1])
    hi, lo = float(seg.max()), float(seg.min())
    last = seg.index[-1]
    d_hi = (last - seg.idxmax()).days
    d_lo = (last - seg.idxmin()).days
    return [round(hi, 4), int(d_hi), round(lo, 4), int(d_lo),
            round(cur / hi - 1, 4), round(cur / lo - 1, 4)]


def trend_series(s: pd.Series, n: int = 252) -> dict | None:
    """近 N 个交易日（默认 1 年）净值走势（首日=100），附 3 个时间刻度。"""
    seg = s.iloc[-n:]
    if len(seg) < 20:
        return None
    base = float(seg.iloc[0])
    if base <= 0:
        return None
    vals = [round(float(v) / base * 100, 1) for v in seg.values]
    idx = seg.index
    return {"v": vals,
            "d": [str(idx[0].date()), str(idx[len(idx) // 2].date()),
                  str(idx[-1].date())]}


def windows_for(nav: dict, codes: list[str]) -> dict:
    out = {}
    for code in codes:
        s = nav.get(code)
        if s is None or len(s) < 60:
            continue
        item = {}
        w1 = window_stats(s, 252)
        w2 = window_stats(s, 504)
        if w1:
            item["w1"] = w1
        if w2:
            item["w2"] = w2
        r1 = price_range(s, 252)
        r3 = price_range(s, 756)
        if r1:
            item["r1"] = r1
        if r3:
            item["r3"] = r3
        item["last"] = str(s.index[-1].date())
        item["nav"] = round(float(s.iloc[-1]), 4)
        out[code] = item
    return out


# ---------------------------------------------------------------- 同类分位

def add_peer_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """按 fund_type 分组算分位；组内样本不足时退回全池。"""
    df = df.copy()
    df["grp"] = df["fund_type"].fillna("未分类")
    specs = {
        "p_sharpe": ("sharpe", True),
        "p_sharpe1": ("sharpe1", True),
        "p_ret1": ("ret1", True),
        "p_ret2": ("ret2", True),
        "p_vol1": ("vol1", False),
    }
    for col, (src, higher_better) in specs.items():
        df[col] = np.nan
        for _, idx in df.groupby("grp").groups.items():
            sub = df.loc[idx, src]
            if sub.notna().sum() >= 20:
                df.loc[idx, col] = sub.rank(pct=True, ascending=higher_better) * 100
        rest = df[col].isna() & df[src].notna()
        if rest.any():
            df.loc[rest, col] = df.loc[rest, src].rank(
                pct=True, ascending=higher_better) * 100
    return df


# ---------------------------------------------------------------- 行业配置

def load_industry(codes: list[str]) -> dict:
    cache = {}
    if os.path.exists(INDUSTRY_CACHE):
        try:
            with open(INDUSTRY_CACHE, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:  # noqa: BLE001
            cache = {}

    todo = [c for c in codes if c not in cache]
    if todo:
        try:
            import akshare as ak
        except Exception:  # noqa: BLE001
            ak = None
        fn = getattr(ak, "fund_portfolio_industry_allocation_em", None) if ak else None
        if fn is None:
            print("      akshare 缺少行业配置接口，跳过行业数据")
        else:
            year = str(datetime.now().year)
            for i, code in enumerate(todo, 1):
                rec = None
                for y in (year, str(int(year) - 1)):
                    try:
                        df = fn(symbol=code, date=y)
                    except Exception:  # noqa: BLE001
                        continue
                    if df is None or df.empty:
                        continue
                    df = df[df["行业类别"].astype(str) != "合计"]
                    if df.empty:
                        continue
                    # 只保留最新报告期，避免多期数据混排
                    dates = df["截止时间"].astype(str)
                    date = dates.max()
                    df = df[dates == date].sort_values("占净值比例", ascending=False)
                    if df.empty:
                        continue
                    rec = {
                        "top": [[str(r["行业类别"]), round(float(r["占净值比例"]), 2)]
                                for _, r in df.head(3).iterrows()],
                        "conc": round(float(df["占净值比例"].head(3).sum()), 2),
                        "date": date,
                    }
                    break
                cache[code] = rec
                if i % 20 == 0:
                    print(f"      行业配置 {i}/{len(todo)}", flush=True)
                time.sleep(0.3)
            os.makedirs(CACHE, exist_ok=True)
            tmp = INDUSTRY_CACHE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            os.replace(tmp, INDUSTRY_CACHE)
    return cache


# ---------------------------------------------------------------- 场外估值

def load_valuation(codes: set[str]) -> dict:
    """东财全市场估算净值快照，仅用于场外基金的“盘中估算”。"""
    try:
        import akshare as ak
    except Exception:  # noqa: BLE001
        return {}
    fn = getattr(ak, "fund_value_estimation_em", None)
    if fn is None:
        return {}
    out = {}
    for symbol in ("全部", "股票型", "混合型", "指数型"):
        try:
            df = fn(symbol=symbol)
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        cols = {c: str(c) for c in df.columns}
        c_code = next((c for c in df.columns if "基金代码" in cols[c]), None)
        c_est = next((c for c in df.columns if "估算值" in cols[c] and "增长" not in cols[c]), None)
        c_chg = next((c for c in df.columns if "估算增长率" in cols[c]), None)
        c_nav = next((c for c in df.columns if "单位净值" in cols[c]), None)
        c_date = next((c for c in df.columns if "估算数据" in cols[c]), None)
        if c_code is None:
            continue
        day = None
        if c_date is not None:
            m = str(c_date).split("-估算")[0]
            day = m
        for _, r in df.iterrows():
            code = str(r[c_code]).zfill(6)
            if code not in codes or code in out:
                continue
            out[code] = [num(r.get(c_est), 4), num(r.get(c_chg), 2),
                         num(r.get(c_nav), 4), day]
    return out


# ---------------------------------------------------------------- 当日预测

def load_intraday() -> dict | None:
    if not os.path.exists(INTRADAY):
        return None
    try:
        with open(INTRADAY, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"      14:30 快照读取失败：{exc}")
        return None


def load_index_daily(secid: str = "1.000300",
                     cache_name: str = "idx300.json") -> pd.Series | None:
    """指数日线收盘（东财 kline），缓存 12 小时；失败时退回旧缓存。"""
    cache_path = os.path.join(CACHE, cache_name)
    data = None
    fresh = (os.path.exists(cache_path)
             and time.time() - os.path.getmtime(cache_path) < 12 * 3600)
    if fresh:
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            data = None
    if data is None:
        try:
            import requests
            r = requests.get(
                "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                params={"secid": secid, "fields1": "f1,f2,f3,f4,f5,f6",
                        "fields2": "f51,f53", "klt": 101, "fqt": 0,
                        "end": "20500101", "lmt": 700,
                        "ut": "fa5fd1943c7b386f172d6893dbfba10b"},
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
                timeout=12)
            klines = ((r.json() or {}).get("data") or {}).get("klines") or []
            rows = []
            for k in klines:
                p = str(k).split(",")
                if len(p) >= 2:
                    rows.append([p[0], float(p[1])])
            if rows:
                data = rows
                os.makedirs(CACHE, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            print(f"      {cache_name} 日线获取失败：{exc}")
        if data is None and os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:  # noqa: BLE001
                data = None
    if not data:
        return None
    s = pd.Series({pd.Timestamp(d): float(v) for d, v in data}).sort_index()
    return s


def betas_for(nav: dict, codes: list[str], idx: pd.Series | None) -> dict:
    """相对上证指数的日收益 β（近 250 日），用于缺失盘中估值时的近似。"""
    if idx is None or idx.empty:
        return {}
    ir = idx.pct_change().dropna()
    out = {}
    for code in codes:
        s = nav.get(code)
        if s is None or len(s) < 120:
            continue
        j = pd.concat([s.pct_change().rename("f"), ir.rename("m")],
                      axis=1, join="inner").dropna().iloc[-250:]
        if len(j) < 60:
            continue
        var = float(j["m"].var())
        if not np.isfinite(var) or var <= 0:
            continue
        beta = float(j["f"].cov(j["m"]) / var)
        if np.isfinite(beta):
            out[code] = round(max(0.2, min(1.8, beta)), 2)
    return out


def market_pred(intraday: dict) -> dict | None:
    """大盘方向打分：上证 + 沪深300 平均涨跌 + 估值广度 + 近 30 分钟斜率。"""
    idx = intraday.get("index") or {}
    idx2 = intraday.get("index2") or {}
    chg1, chg2 = idx.get("chg"), idx2.get("chg")
    chgs = [c for c in (chg1, chg2) if c is not None]
    chg = sum(chgs) / len(chgs) if chgs else None
    br = intraday.get("breadth") or {}
    upr = br.get("upRatio")
    trend = idx.get("trend") or []
    tail = None
    if len(trend) >= 8:
        seg = [p[1] for p in trend[-7:]]
        if seg[0]:
            tail = (seg[-1] / seg[0] - 1) * 100

    score, reasons = 0.0, []
    if chg is not None:
        score += 0.45 * np.tanh(chg / 1.2)
        if chg1 is not None and chg2 is not None:
            reasons.append(f"上证指数 {chg1:+.2f}% · 沪深300 {chg2:+.2f}%")
        else:
            reasons.append(f"大盘 {chg:+.2f}%，日内{'偏强' if chg > 0 else '偏弱'}")
    if upr is not None:
        score += 0.25 * np.tanh((upr - 50) / 25)
        reasons.append(f"指数基金估值上涨 {br.get('up')} 只 / 下跌 {br.get('down')} 只"
                       f"（上涨占比 {upr:.0f}%）")
    if tail is not None:
        score += 0.30 * np.tanh(tail / 0.35)
        reasons.append(f"最近 30 分钟指数{'回升' if tail >= 0 else '回落'} {tail:+.2f}%")
    if not reasons:
        return None

    score = float(np.clip(score, -1, 1))
    if score >= 0.1:
        direction = "偏涨"
    elif score <= -0.1:
        direction = "偏跌"
    else:
        direction = "震荡"
    conf = int(round(min(85, 50 + abs(score) * 45)))
    return {"dir": direction, "score": round(score, 2), "conf": conf,
            "reasons": reasons, "tail": None if tail is None else round(tail, 2)}


def fund_pred(mkt: dict, chg: float | None, mom5: float | None,
              approx: bool) -> dict | None:
    if not mkt:
        return None
    score = 0.5 * mkt["score"]
    reasons = []
    if chg is not None:
        score += 0.3 * np.tanh(chg / 1.2)
        reasons.append(f"14:30 估值 {chg:+.2f}%" + ("（β近似）" if approx else "（实时/估算）"))
    if mom5 is not None:
        score += 0.2 * np.tanh(mom5 / 2.5)
        reasons.append(f"近 5 日 {mom5:+.2f}%")
    reasons.extend(mkt["reasons"][:2])
    score = float(np.clip(score, -1, 1))
    if score >= 0.1:
        direction = "偏涨"
    elif score <= -0.1:
        direction = "偏跌"
    else:
        direction = "震荡"
    conf = int(round(min(85, 50 + abs(score) * 45)))
    return {"dir": direction, "score": round(score, 2), "conf": conf,
            "reasons": reasons[:4], "chg": chg, "approx": approx}


def price_position(s: pd.Series | None, chg: float | None) -> dict | None:
    """把 14:30 估值涨跌套到最新净值上，算近 N 日区间分位。"""
    if s is None or len(s) < 30:
        return None
    adj = float(s.iloc[-1]) * (1 + (chg or 0) / 100)
    windows = {}
    for w in (20, 60, 120, 250):
        seg = s.iloc[-w:]
        if len(seg) < max(20, int(w * 0.6)):
            continue
        lo, hi = float(seg.min()), float(seg.max())
        p = (adj - lo) / (hi - lo) * 100 if hi > lo else 50.0
        windows[str(w)] = [round(p, 1), round(lo, 4), round(hi, 4)]
    if not windows:
        return None
    return {"adj": round(adj, 4), "w": windows}


def make_advice(pos: dict | None, pdir: str | None, approx: bool) -> dict | None:
    if not pos:
        return None
    days = "120" if "120" in pos["w"] else max(pos["w"], key=int)
    p, lo, hi = pos["w"][days]
    low = p <= 15
    high = p >= 85

    if pdir == "偏涨":
        if low:
            bucket, text = "分批买入", "处于区间低位且当日偏强，可小额分批建仓；不追高。"
        elif high:
            bucket, text = "分批止盈", "位置偏高且当日偏强，建议分批止盈而非加仓。"
        elif p <= 40:
            bucket, text = "回调分批", "中低位、当日偏强，可逢回调分批，单次仓位不宜大。"
        else:
            bucket, text = "持有", "中位偏强，已有仓位可持有，暂不追涨。"
    elif pdir == "偏跌":
        if high:
            bucket, text = "减仓止盈", "位置偏高且当日偏弱，优先减仓/止盈。"
        elif low:
            bucket, text = "等待企稳", "已在区间低位，但当日偏弱，等企稳信号再分批。"
        elif p <= 40:
            bucket, text = "暂缓买入", "中低位但当日偏弱，暂缓加仓，观察次日是否止跌。"
        else:
            bucket, text = "控制仓位", "中位偏弱，控制仓位，等方向明确。"
    else:
        if low:
            bucket, text = "小仓试买", "区间低位、方向不明，可极小额试仓，留足余量。"
        elif high:
            bucket, text = "逢高减仓", "区间高位、方向不明，逢高减仓更稳。"
        elif p <= 40:
            bucket, text = "小额分批", "中低位震荡，可小额分批，关注大盘方向。"
        else:
            bucket, text = "观望", "中位震荡，观望为主，等回踩或突破。"

    note = f"14:30 价约 {pos['adj']:.4f}，处于近 {days} 日区间 {p:.0f}% 分位（低 {lo:.4f} / 高 {hi:.4f}）"
    if p <= 10:
        note += f"，接近近 {days} 日最低"
    if approx:
        note += "；该基金无盘中估值，当日涨跌按 β 近似"
    return {"bucket": bucket, "text": text, "note": note, "p": p,
            "days": int(days), "lo": lo, "hi": hi, "adj": pos["adj"]}



def build_records():
    t = read("tiers.csv")
    if t.empty:
        sys.exit("缺少 output/tiers.csv，请先运行 fund_mvp.py")
    t["fund_code"] = t["fund_code"].str.zfill(6)
    m = read("fund_metrics.csv")
    g = read("gate2_report.csv")
    s = read("style_box.csv")
    for df in (m, g, s):
        if not df.empty:
            df["fund_code"] = df["fund_code"].str.zfill(6)
    if not m.empty:
        keep = ["fund_code", "max_drawdown", "calmar", "sharpe", "ir", "win_rate", "fund_type"]
        t = t.merge(m[[c for c in keep if c in m.columns]], on="fund_code", how="left")
    if not g.empty:
        t = t.merge(g[["fund_code", "aum_yi", "drift_level"]], on="fund_code", how="left")
    if not s.empty:
        t = t.merge(s[["fund_code", "style_quadrant", "style_r2"]], on="fund_code", how="left")

    codes = t["fund_code"].tolist()
    nav = load_nav_store()
    win = windows_for(nav, codes)
    print(f"      净值窗口：{len(win)}/{len(codes)} 只")

    t["ret1"] = [win.get(c, {}).get("w1", [None] * 3)[0] for c in codes]
    t["vol1"] = [win.get(c, {}).get("w1", [None] * 3)[1] for c in codes]
    t["sharpe1"] = [win.get(c, {}).get("w1", [None] * 3)[2] for c in codes]
    t["ret2"] = [win.get(c, {}).get("w2", [None] * 3)[0] for c in codes]
    t = add_peer_percentiles(t)

    pool = t[t["tier"].isin(["S", "A", "B", "C"])]["fund_code"].tolist()
    pool_set = set(pool)
    industry = load_industry(pool)
    valuations = load_valuation(set(pool))

    intraday = load_intraday()
    mkt = market_pred(intraday) if intraday else None
    betas = {}
    if intraday and (intraday.get("index") or {}).get("chg") is not None:
        betas = betas_for(nav, pool, load_index_daily())
        print(f"      当日预测：快照 {intraday.get('date')} {intraday.get('time')}，"
              f"β 覆盖 {len(betas)} 只" + (f"，大盘{mkt['dir']}" if mkt else ""))
    intra_funds = (intraday or {}).get("funds") or {}
    idx_chg = ((intraday or {}).get("index2") or {}).get("chg")
    if idx_chg is None:
        idx_chg = ((intraday or {}).get("index") or {}).get("chg")

    recs = []
    for _, r in t.iterrows():
        code = r["fund_code"]
        w = win.get(code, {})
        s = nav.get(code)
        chg, approx = None, False
        if code in intra_funds:
            chg = intra_funds[code]
        elif valuations.get(code) and valuations[code][1] is not None:
            chg = valuations[code][1]
        elif idx_chg is not None and code in betas:
            chg, approx = round(betas[code] * idx_chg, 2), True
        mom5 = None
        if s is not None and len(s) >= 6:
            mom5 = round(float(s.iloc[-1] / s.iloc[-6] - 1) * 100, 2)
        pred = fund_pred(mkt, chg, mom5, approx) if (mkt and code in pool_set) else None
        pos = price_position(s, chg) if code in pool_set else None
        advice = make_advice(pos, pred["dir"] if pred else None, approx)
        item = {
            "code": code, "name": str(r.get("fund_name") or ""),
            "tier": r["tier"], "score": num(r.get("score"), 1) or 0,
            "powers": [num(r.get(k), 0) or 0 for k in DIM_KEYS],
            "ann": num(r.get("ann_return"), 4), "excess": num(r.get("ref_excess_pp"), 1),
            "mdd": num(r.get("max_drawdown"), 4), "calmar": num(r.get("calmar"), 2),
            "sharpe": num(r.get("sharpe"), 2), "aum": num(r.get("aum_yi"), 1),
            "quad": (str(r["style_quadrant"]) if pd.notna(r.get("style_quadrant")) else ""),
            "drift": (str(r["drift_level"]) if pd.notna(r.get("drift_level")) else ""),
            "company": ("" if pd.isna(r.get("company")) else str(r.get("company"))),
            "type": ("" if pd.isna(r.get("fund_type")) else str(r.get("fund_type"))) or "未分类",
            "w1": w.get("w1"), "w2": w.get("w2"),
            "r1": w.get("r1"), "r3": w.get("r3"),
            "last": w.get("last"), "nav": w.get("nav"),
            "pr": [num(r.get("p_sharpe"), 1), num(r.get("p_sharpe1"), 1),
                   num(r.get("p_ret1"), 1), num(r.get("p_ret2"), 1),
                   num(r.get("p_vol1"), 1)],
            "ind": industry.get(code),
            "val": valuations.get(code),
            "trend": (trend_series(s) if code in pool_set and s is not None else None),
            "pred": pred, "advice": advice,
        }
        recs.append(item)
    recs.sort(key=lambda x: (TIERS.index(x["tier"]), -x["score"]))
    return recs, intraday, mkt


def index_year(idx: pd.Series | None) -> dict | None:
    """上证指数近 1 年走势（首日=100），用于顶部右侧曲线。"""
    if idx is None or len(idx) < 60:
        return None
    seg = idx.iloc[-252:]
    base = float(seg.iloc[0])
    if base <= 0:
        return None
    vals = [round(float(v) / base * 100, 1) for v in seg.values]
    return {"v": vals,
            "d": [str(seg.index[0].date()), str(seg.index[len(seg) // 2].date()),
                  str(seg.index[-1].date())],
            "chg": round(float(seg.iloc[-1]) / base - 1, 4)}


def main():
    recs, intraday, mkt = build_records()
    counts = {k: sum(1 for r in recs if r["tier"] == k) for k in TIERS}
    gaps = [r["ann"] - r["excess"] / 100 for r in recs
            if r["ann"] is not None and r["excess"] is not None]
    ref = round(sum(gaps) / len(gaps), 4) if gaps else 0.0
    snap = None
    if intraday:
        idx_now = intraday.get("index") or {}
        idx_now["year"] = index_year(load_index_daily("1.000001", "idx001.json"))
        idx_now2 = intraday.get("index2") or {}
        idx_now2["year"] = index_year(load_index_daily())
        snap = {
            "date": intraday.get("date"), "time": intraday.get("time"),
            "stamp": intraday.get("stamp"),
            "index": idx_now,
            "index2": idx_now2,
            "slots": intraday.get("slots") or [],
            "breadth": intraday.get("breadth") or {},
            "pred": mkt,
            "covered": sum(1 for r in recs if r.get("pred")),
        }
    payload = {"funds": recs, "counts": counts, "tiers": TIERS, "ref": ref,
               "tierDesc": TIER_DESC, "dims": DIM_NAMES,
               "intraday": snap,
               "builtAt": datetime.now().strftime("%Y-%m-%d %H:%M")}

    tpl = open(os.path.join(BASE, "dashboard_template.html"), encoding="utf-8").read()
    html = tpl.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    dst = os.path.join(OUT, "tier_dashboard.html")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成 {dst}（{len(recs)} 只基金，{os.path.getsize(dst) / 1024:.0f} KB）")
    print("分级分布：" + "  ".join(f"{k}={counts[k]}" for k in TIERS if counts[k]))


if __name__ == "__main__":
    main()
