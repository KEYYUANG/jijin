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


def trend_series(s: pd.Series, n: int = 120) -> dict | None:
    """近 N 个交易日净值走势（首日=100），附 3 个时间刻度。"""
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


# ---------------------------------------------------------------- 组装

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

    recs = []
    for _, r in t.iterrows():
        code = r["fund_code"]
        w = win.get(code, {})
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
            "trend": (trend_series(nav[code]) if code in pool_set and code in nav else None),
        }
        recs.append(item)
    recs.sort(key=lambda x: (TIERS.index(x["tier"]), -x["score"]))
    return recs


def main():
    recs = build_records()
    counts = {k: sum(1 for r in recs if r["tier"] == k) for k in TIERS}
    gaps = [r["ann"] - r["excess"] / 100 for r in recs
            if r["ann"] is not None and r["excess"] is not None]
    ref = round(sum(gaps) / len(gaps), 4) if gaps else 0.0
    payload = {"funds": recs, "counts": counts, "tiers": TIERS, "ref": ref,
               "tierDesc": TIER_DESC, "dims": DIM_NAMES,
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
