# -*- coding: utf-8 -*-
"""基金 MVP：第一道门（相对阈值）+ 风格箱（四因子回归）。

设计要点
  1. 第一道门默认使用“同类相对分位”，而不是文档里的绝对阈值（实测绝对阈值会得到空池）。
  2. 数据双源 + 本地缓存 + 退避重试：指数用新浪为主、东财为备；基金净值走东财并落盘缓存。
  3. 风格箱用国证四风格指数（大盘价值/大盘成长/小盘价值/小盘成长）做滚动回归，
     按季度快照统计风格漂移。

用法
  python fund_mvp.py --limit 30
  python fund_mvp.py --limit 100 --years 5
  python fund_mvp.py --codes 000001,110011 --mode absolute
  python fund_mvp.py --limit 40 --total-value 100000 --new-money 5000
  python fund_mvp.py --limit 40 --holdings holdings.csv --total-value 100000
  python fund_mvp.py --limit 40 --force-review      # 立即复核观察池（不等一个季度）
  python fund_mvp.py --limit 40 --reset-state       # 清空观察池状态重新开始

输出（默认 output/）
  fund_metrics.csv      指标与分位
  gate1_report.csv      第一道门判定
  style_box.csv         最新风格箱 + 漂移等级
  style_history.csv     风格箱季度快照
  gate2_report.csv      第二道门：经理稳定性/规模/风格漂移
  fund_pool.csv         核心-卫星分层与目标权重
  rebalance_actions.csv 再平衡动作（5pp 偏离 + 卫星盈利 20% 落袋）
  watchlist.csv         观察池（含跨期状态：watching/reactivated/retired/expired）
  watchlist_events.csv  本期状态机事件（entered/review_keep/reactivated/retired/lock_expired）
  gate3_report.csv      第三道门约束校验（分层比例/集中度/现金）
  new_money_plan.csv    新增资金分档投入（--new-money）
  data_failures.csv     取数失败清单
  summary.txt           中文汇总
  state/watchlist.json  跨期状态（下次运行自动续接）
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import akshare as ak
import numpy as np
import pandas as pd
import requests

TRADING_DAYS = 252

# 国证四风格指数（新浪代码）：大盘价值/大盘成长/小盘价值/小盘成长
STYLE_INDEXES = {
    "big_value": "sz399373",
    "big_growth": "sz399374",
    "small_value": "sz399377",
    "small_growth": "sz399378",
}
BENCH_HS300 = "sh000300"
BENCH_BOND = "sh000012"  # 上证国债指数，作为中债综合财富的免费代理

EQUITY_TYPES = ["股票型", "混合型-偏股", "混合型-灵活", "混合型-平衡"]
INDEX_TYPES = ["指数型-股票"]


@dataclass
class Config:
    start: str | None = None
    years: float = 5.0
    mode: str = "relative"          # relative | absolute
    limit: int = 50
    codes: list[str] = field(default_factory=list)
    include_index: bool = False
    min_days: int = 500
    prescreen_aum: float = 2.0        # 预筛：规模下限（亿元）
    prescreen_years: float = 3.0      # 预筛：成立年限下限
    no_prescreen: bool = False
    risk_free: float = 0.02
    sleep: float = 0.25
    retries: int = 3
    cache_dir: str = "cache"
    outdir: str = "output"
    refresh: bool = False
    style_window: int = 250         # 回归窗口（约一年）
    style_step: int = 63            # 快照间隔（约一个季度）
    style_periods: int = 8          # 取最近 8 个快照
    drift_mild: int = 2
    drift_significant: int = 3
    # 第二道门
    min_manager_tenure: float = 3.0     # 现任经理任职年限下限（年）
    max_manager_changes: int = 2        # 近 3 年变更次数上限
    max_manager_products: int = 5       # 同一经理在管产品数上限
    core_tenure: float = 5.0            # 核心仓经理任职年限
    core_aum_yi: float = 10.0           # 核心仓规模下限（亿元）
    # 第三道门
    core_ratio: float = 0.65
    satellite_ratio: float = 0.35
    satellite_cap: float = 0.10
    quadrant_cap: float = 0.40
    company_cap: float = 0.30
    rebalance_margin: float = 0.05
    profit_take: float = 0.20
    total_value: float = 0.0            # 组合总市值，>0 时换算金额
    holdings: str | None = None
    new_money: float = 0.0
    # 观察池状态机
    state_dir: str = "state"
    review_interval_days: int = 80      # 评估周期（约一个季度）
    max_watch_periods: int = 2          # 连续 2 期未复活 -> 永久退役
    revive_calmar_ratio: float = 0.8    # Calmar 回升至入库值 80%
    revive_tenure: float = 1.0          # 复活要求经理任职满 1 年
    revive_scale_band: float = 0.5      # 规模企稳：单季变动幅度不超过 ±50%
    retire_lock_months: int = 6         # 退役后 6 个月内不得重新提交
    failure_ttl_days: int = 30          # 取数失败缓存有效期（天）
    no_tier_refresh: bool = False       # 关闭分级刷新（全部走缓存）
    force_review: bool = False
    force_rebalance: bool = False
    clear_failures: bool = False


# ---------------------------------------------------------------- 数据层

def _retry(fn, *args, retries=3, sleep=2.0, **kwargs):
    last = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < retries - 1:
                time.sleep(sleep * (i + 1))
    raise last  # type: ignore[misc]


def _cached_csv(path: str, refresh: bool, loader) -> pd.DataFrame:
    if (not refresh) and os.path.exists(path):
        return pd.read_csv(path)
    df = loader()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


_INDEX_MEM: dict[str, pd.Series] = {}


def fetch_index(symbol: str, cfg: Config) -> pd.Series:
    """指数日线：新浪为主，东财为备（同进程内只读一次）。"""
    if not cfg.refresh and symbol in _INDEX_MEM:
        return _INDEX_MEM[symbol]
    cache = os.path.join(cfg.cache_dir, f"idx_{symbol}.csv")

    def load() -> pd.DataFrame:
        try:
            df = _retry(ak.stock_zh_index_daily, symbol=symbol,
                        retries=cfg.retries, sleep=2.0)
        except Exception:
            # 备用源：东财（push2，可能限流）
            em_symbol = symbol[2:]
            df = _retry(ak.index_zh_a_hist, symbol=em_symbol, period="daily",
                        retries=cfg.retries, sleep=3.0)
            df = df.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]]
        return df[["date", "close"]]

    df = _cached_csv(cache, cfg.refresh, load)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].astype(float).sort_index()


def _nav_series(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns or "nav" not in df.columns:
        return pd.Series(dtype=float)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    s = (df.drop_duplicates("date").set_index("date")["nav"]
         .astype(float).sort_index())
    return s[s > 0]


_NAV_STORE: dict[str, pd.Series] | None = None


def nav_store_path(cfg: Config) -> str:
    return os.path.join(cfg.cache_dir, "nav_store.pkl")


def load_nav_store(cfg: Config) -> dict[str, pd.Series]:
    """单文件净值缓存：优先读 pkl，首次从逐只 CSV 迁移。"""
    global _NAV_STORE
    if _NAV_STORE is not None:
        return _NAV_STORE

    path = nav_store_path(cfg)
    if os.path.exists(path):
        with open(path, "rb") as f:
            _NAV_STORE = pickle.load(f)
        return _NAV_STORE

    files = glob.glob(os.path.join(cfg.cache_dir, "nav_*.csv"))
    store: dict[str, pd.Series] = {}

    def read_one(p: str):
        code = os.path.basename(p)[len("nav_"):-len(".csv")]
        try:
            s = _nav_series(pd.read_csv(p))
            return code, s if len(s) else None
        except Exception:  # noqa: BLE001
            return code, None

    if files:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for code, s in pool.map(read_one, files):
                if s is not None:
                    store[code] = s
        print(f"      合并缓存：迁移 {len(store)}/{len(files)} 只净值到 nav_store.pkl",
              flush=True)

    _NAV_STORE = store
    save_nav_store(cfg)
    return _NAV_STORE


def save_nav_store(cfg: Config) -> None:
    if _NAV_STORE is None:
        return
    os.makedirs(cfg.cache_dir, exist_ok=True)
    tmp = nav_store_path(cfg) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(_NAV_STORE, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, nav_store_path(cfg))


def fetch_nav(code: str, cfg: Config, force: bool = False) -> pd.Series:
    """基金累计净值（含分红再投）。命中内存/单文件缓存则直接返回。"""
    if _NAV_STORE is not None and not cfg.refresh and not force and code in _NAV_STORE:
        return _NAV_STORE[code]

    def load() -> pd.DataFrame:
        try:
            df = _retry(ak.fund_open_fund_info_em, symbol=code,
                        indicator="累计净值走势", retries=cfg.retries, sleep=1.5)
            df = df.rename(columns={"净值日期": "date", "累计净值": "nav"})
        except Exception:
            df = _retry(ak.fund_open_fund_info_em, symbol=code,
                        indicator="单位净值走势", retries=cfg.retries, sleep=1.5)
            df = df.rename(columns={"净值日期": "date", "单位净值": "nav"})
        return df[["date", "nav"]]

    s = _nav_series(load())
    if _NAV_STORE is not None and len(s):
        _NAV_STORE[code] = s
    return s


EM_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://fundf10.eastmoney.com/"}


def _http_get(url: str, cfg: Config, timeout: int = 20) -> str:
    def load() -> str:
        r = requests.get(url, headers=EM_HEADERS, timeout=timeout)
        r.raise_for_status()
        r.encoding = "utf-8"
        return r.text

    return _retry(load, retries=cfg.retries, sleep=2.0)


def fetch_manager_history(code: str, cfg: Config) -> pd.DataFrame:
    """历任基金经理（天天基金 F10 页面，表头：起始期/截止期/基金经理）。"""
    cache = os.path.join(cfg.cache_dir, f"mgr_{code}.csv")
    if (not cfg.refresh) and os.path.exists(cache):
        return pd.read_csv(cache)

    html = _http_get(f"https://fundf10.eastmoney.com/jjjl_{code}.html", cfg)
    hist = pd.DataFrame(columns=["start", "end", "manager", "tenure_days", "ret"])
    for t in pd.read_html(io.StringIO(html)):
        cols = [str(c) for c in t.columns]
        if "起始期" in cols and "基金经理" in cols:
            hist = t.rename(columns={"起始期": "start", "截止期": "end",
                                     "基金经理": "manager", "任职期间": "tenure_days",
                                     "任职回报": "ret"})
            hist = hist[["start", "end", "manager", "tenure_days", "ret"]]
            break
    hist.to_csv(cache, index=False, encoding="utf-8-sig")
    return hist


def fetch_scale_history(code: str, cfg: Config) -> pd.DataFrame:
    """规模变动（天天基金 F10 数据接口）：日期/期末净资产(亿元)/净资产变动率。"""
    cache = os.path.join(cfg.cache_dir, f"scale_{code}.csv")
    if (not cfg.refresh) and os.path.exists(cache):
        return pd.read_csv(cache)

    text = _http_get(
        f"https://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=gmbd&code={code}&rt=0.1", cfg)
    m = re.search(r'content:"(.*?)",', text, re.S)
    out = pd.DataFrame(columns=["date", "nav_yi", "chg"])
    if m:
        t = pd.read_html(io.StringIO(m.group(1).replace('\\"', '"')))[0]
        if t.shape[1] >= 6:
            out = pd.DataFrame({
                "date": t.iloc[:, 0].astype(str),
                "nav_yi": pd.to_numeric(t.iloc[:, 4], errors="coerce"),
                "chg": pd.to_numeric(
                    t.iloc[:, 5].astype(str).str.replace("%", "", regex=False),
                    errors="coerce") / 100.0,
            })
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def fetch_fund_snapshot(code: str, cfg: Config) -> dict:
    """最新规模/公司（天天基金移动端基础信息接口）。"""
    cache = os.path.join(cfg.cache_dir, f"info_{code}.csv")
    if (not cfg.refresh) and os.path.exists(cache):
        row = pd.read_csv(cache).iloc[0].to_dict()
        return row

    url = ("https://fundmobapi.eastmoney.com/FundMNewApi/FundMNNBasicInformation"
           f"?FCODE={code}&deviceid=1&plat=Android&product=EFund&version=6.3.8")
    row = {"fund_code": code, "company": None, "aum_yi": np.nan, "estab_date": None}
    for attempt in range(2):
        try:
            data = requests.get(url, headers=EM_HEADERS, timeout=12).json().get("Datas") or {}
            row["company"] = data.get("JJGS")
            nav = data.get("ENDNAV")
            row["aum_yi"] = float(nav) / 1e8 if nav else np.nan
            row["estab_date"] = data.get("ESTABDATE")
            break
        except Exception:  # noqa: BLE001
            if attempt == 0:
                time.sleep(0.5)
    pd.DataFrame([row]).to_csv(cache, index=False, encoding="utf-8-sig")
    return row


def load_manager_map(cfg: Config) -> dict[str, int]:
    """经理姓名 -> 在管产品数。"""
    cache = os.path.join(cfg.cache_dir, "fund_manager.csv")
    try:
        df = (_cached_csv(cache, cfg.refresh, lambda: _retry(
            ak.fund_manager_em, retries=cfg.retries, sleep=2.0)))
        df = df.dropna(subset=["姓名", "现任基金代码"])
        return df.groupby("姓名")["现任基金代码"].nunique().to_dict()
    except Exception:  # noqa: BLE001
        return {}


def _failure_path(cfg: Config) -> str:
    return os.path.join(cfg.state_dir, "failures.json")


def load_failures(cfg: Config) -> dict:
    path = _failure_path(cfg)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_failures(state: dict, cfg: Config) -> None:
    os.makedirs(cfg.state_dir, exist_ok=True)
    with open(_failure_path(cfg), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def is_blacklisted(entry: dict | None, cfg: Config, today: pd.Timestamp) -> bool:
    """失败缓存是否仍在有效期内（默认 30 天，到期自动重试）。"""
    if not entry:
        return False
    try:
        last = pd.Timestamp(str(entry.get("last_try", "1970-01-01")))
    except (ValueError, TypeError):
        return False
    return (today - last).days < cfg.failure_ttl_days


def prescreen(uni: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """规模 + 成立年限预筛：先批量拉基础信息，再决定是否取净值。"""
    codes = [str(c) for c in uni["基金代码"]]
    path_of = lambda c: os.path.join(cfg.cache_dir, f"info_{c}.csv")
    todo = [c for c in codes if cfg.refresh or not os.path.exists(path_of(c))]
    print(f"      预筛：需拉取 {len(todo)} 只基础信息（缓存 {len(codes) - len(todo)}）", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch_fund_snapshot, c, cfg) for c in todo]
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception:  # noqa: BLE001
                pass
            if done % 200 == 0:
                print(f"      基础信息 {done}/{len(todo)}", flush=True)

    meta: dict[str, dict] = {}
    for c in codes:
        aum, estab = np.nan, None
        if os.path.exists(path_of(c)):
            try:
                row = pd.read_csv(path_of(c)).iloc[0]
                aum = float(row["aum_yi"]) if pd.notna(row["aum_yi"]) else np.nan
                estab = row.get("estab_date")
            except Exception:  # noqa: BLE001
                pass
        meta[c] = {"aum_yi": aum, "estab_date": estab}

    keep = []
    for c in codes:
        m = meta[c]
        if pd.notna(m["aum_yi"]) and m["aum_yi"] < cfg.prescreen_aum:
            continue
        if cfg.prescreen_years > 0 and pd.notna(m["estab_date"]):
            try:
                age = (pd.Timestamp.today() - pd.Timestamp(str(m["estab_date"]))).days / 365.25
                if age < cfg.prescreen_years:
                    continue
            except (ValueError, TypeError):
                pass
        keep.append(c)

    before = len(uni)
    out = uni[uni["基金代码"].astype(str).isin(set(keep))].copy()
    print(f"      预筛后 {len(out)}/{before} 只"
          f"（规模≥{cfg.prescreen_aum}亿、成立≥{cfg.prescreen_years}年）", flush=True)
    return out


def build_benchmarks(cfg: Config) -> dict[str, pd.Series]:
    hs300 = fetch_index(BENCH_HS300, cfg)
    bond = fetch_index(BENCH_BOND, cfg)
    idx = hs300.index.intersection(bond.index)
    mixed = ((0.6 * hs300.loc[idx].pct_change(fill_method=None)
              + 0.4 * bond.loc[idx].pct_change(fill_method=None))
             .dropna())
    mixed = (1 + mixed).cumprod()
    return {"hs300": hs300, "mixed": mixed}


def pick_benchmark(fund_type: str, cfgs: dict[str, pd.Series]) -> pd.Series:
    ft = "" if fund_type is None else str(fund_type)
    if ft.startswith("混合型"):
        return cfgs["mixed"]
    return cfgs["hs300"]


# ---------------------------------------------------------------- 指标层

def compute_metrics(nav: pd.Series, bench: pd.Series, start: pd.Timestamp,
                    rf: float) -> dict | None:
    s = nav[nav.index >= start]
    b = bench[bench.index >= start]
    if len(s) < 2:
        return None
    r = s.pct_change(fill_method=None).dropna()
    if len(r) < 100:
        return None

    years = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    ann = float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    mdd = float((s / s.cummax() - 1).min())
    sharpe = float((ann - rf) / vol) if vol else np.nan
    calmar = float(ann / abs(mdd)) if mdd else np.nan

    common = r.index.intersection(b.index)
    if len(common) > 100:
        excess = r.loc[common] - b.loc[common].pct_change(fill_method=None).dropna()
        ir = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) if excess.std() else np.nan
    else:
        ir = np.nan

    mf = s.resample("ME").last().pct_change(fill_method=None).dropna()
    mb = b.resample("ME").last().pct_change(fill_method=None).dropna()
    mi = mf.index.intersection(mb.index)
    win = float((mf.loc[mi] > mb.loc[mi]).mean()) if len(mi) >= 6 else np.nan

    return {
        "ann_return": ann,
        "vol": vol,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "ir": ir,
        "win_rate": win,
        "days": len(r),
        "first_date": s.index[0].strftime("%Y-%m-%d"),
        "last_date": s.index[-1].strftime("%Y-%m-%d"),
    }


ABSOLUTE_RULES = {
    "max_drawdown": (">=", -0.20),
    "calmar": (">=", 1.0),
    "sharpe": (">=", 1.0),
    "ir": (">=", 0.5),
    "win_rate": (">=", 0.55),
}
RELATIVE_RULES = {  # 分位阈值：越大越好
    "max_drawdown": 50,   # 回撤不差于同类中位数
    "calmar": 70,         # 同类前 30%
    "sharpe": 70,
    "ir": 60,
    "win_rate": 60,
}
PCT_RULE_TEXT = {k: f"P{v}" for k, v in RELATIVE_RULES.items()}


def apply_gate1(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    keys = list(ABSOLUTE_RULES)

    for k in keys:
        out[f"{k}_pct"] = out[k].rank(pct=True) * 100

    abs_pass, abs_reason = [], []
    for _, row in out.iterrows():
        reasons = []
        for k, (op, th) in ABSOLUTE_RULES.items():
            v = row[k]
            ok = (v >= th) if op == ">=" else (v <= th)
            if pd.isna(v) or not ok:
                reasons.append(k)
        abs_pass.append(len(reasons) == 0)
        abs_reason.append(",".join(reasons))
    out["pass_absolute"] = abs_pass
    out["reject_absolute"] = abs_reason

    if mode == "absolute":
        out["gate1_pass"] = out["pass_absolute"]
        out["reject_reason"] = out["reject_absolute"]
    else:
        reasons = []
        for _, row in out.iterrows():
            bad = [k for k, p in RELATIVE_RULES.items()
                   if pd.isna(row[f"{k}_pct"]) or row[f"{k}_pct"] < p]
            reasons.append(",".join(bad))
        out["gate1_pass"] = [r == "" for r in reasons]
        out["reject_reason"] = reasons
    return out


# ---------------------------------------------------------------- 第二道门

def manager_metrics(code: str, mgr_map: dict[str, int], cfg: Config) -> dict:
    """现任经理任职年限、近 3 年变更次数、在管产品数。"""
    empty = {"manager_name": None, "manager_tenure_years": np.nan,
             "manager_changes_3y": np.nan, "manager_product_count": np.nan,
             "manager_data": "unavailable"}
    try:
        hist = fetch_manager_history(code, cfg)
    except Exception:  # noqa: BLE001
        return empty
    if hist.empty:
        return {**empty, "manager_data": "missing"}

    hist = hist.copy()
    hist["start_dt"] = pd.to_datetime(hist["start"], errors="coerce")
    is_now = hist["end"].astype(str).str.contains("至今")
    current = hist[is_now] if is_now.any() else hist.sort_values("start_dt").tail(1)
    names: list[str] = []
    for cell in current["manager"].astype(str):
        names.extend([n for n in re.split(r"[、,，\s]+", cell) if n])
    tenure = np.nan
    if current["start_dt"].notna().any():
        tenure = float((pd.Timestamp.today() - current["start_dt"].min()).days / 365.25)

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=3)
    changes = int((hist["start_dt"] >= cutoff).sum())
    if hist["start_dt"].notna().any() and hist["start_dt"].min() >= cutoff:
        changes = max(changes - 1, 0)  # 成立后首任不算“变更”

    counts = [mgr_map.get(n, np.nan) for n in names]
    products = float(np.nanmax(counts)) if counts and not np.all(np.isnan(counts)) else np.nan
    return {"manager_name": "、".join(names), "manager_tenure_years": tenure,
            "manager_changes_3y": changes, "manager_product_count": products,
            "manager_data": "ok"}


def scale_metrics(hist: pd.DataFrame) -> dict:
    if hist is None or hist.empty:
        return {"aum_yi": np.nan, "aum_chg": np.nan, "scale_quarters": 0}
    h = hist.dropna(subset=["nav_yi"]).head(4)
    if h.empty:
        return {"aum_yi": np.nan, "aum_chg": np.nan, "scale_quarters": 0}
    return {"aum_yi": float(h.iloc[0]["nav_yi"]),
            "aum_chg": float(h.iloc[0]["chg"]) if pd.notna(h.iloc[0]["chg"]) else np.nan,
            "scale_quarters": len(h)}


def evaluate_gate2(style: dict | None, mgr: dict, scale: dict, snap: dict,
                   cfg: Config) -> dict:
    """第二道门：风格与经理稳定性。返回 pass / pass_flag（仅卫星）/ watchlist。"""
    drift = (style or {}).get("drift_level", "unknown")
    tenure = mgr["manager_tenure_years"]
    changes = mgr["manager_changes_3y"]
    products = mgr["manager_product_count"]
    aum = scale["aum_yi"]
    if pd.isna(aum) and pd.notna(snap.get("aum_yi")):
        aum = float(snap["aum_yi"])

    watch = []
    confidence = (style or {}).get("style_confidence", "unknown")
    confirmed = (style or {}).get("drift_confirmed", False)
    if drift == "significant_drift" and confidence != "unclear" and confirmed:
        watch.append("significant_drift")
    if pd.notna(changes) and changes >= cfg.max_manager_changes:
        watch.append(f"manager_changes_{int(changes)}")

    flags = []
    if not watch:
        if drift == "mild_drift":
            flags.append("mild_drift")
        if drift == "significant_drift":
            flags.append("drift_unconfirmed" if not confirmed else "style_unclear")
        if drift == "unknown":
            flags.append("style_missing")
        if pd.isna(tenure):
            flags.append("manager_data_missing")
        elif tenure < cfg.min_manager_tenure:
            flags.append("short_tenure")
        if pd.notna(products) and products > cfg.max_manager_products:
            flags.append("overloaded")
        if pd.notna(scale["aum_chg"]) and (scale["aum_chg"] > 1.0 or scale["aum_chg"] < -0.5):
            flags.append("scale_shock")

    status = "watchlist" if watch else ("pass_flag" if flags else "pass")
    core_ok = (status == "pass"
               and drift == "stable"
               and pd.notna(tenure) and tenure >= cfg.core_tenure
               and pd.notna(aum) and aum > cfg.core_aum_yi)
    return {"gate2_result": status, "gate2_flags": ",".join(flags),
            "watch_reason": ",".join(watch), "core_eligible": bool(core_ok),
            "aum_yi": aum, "aum_chg": scale["aum_chg"],
            "turnover_flag": "unknown"}


# ---------------------------------------------------------------- 第三道门

def _spread(codes: list[str], total: float, cap: float) -> dict[str, float]:
    if not codes:
        return {}
    w = {c: total / len(codes) for c in codes}
    for _ in range(100):
        over = [c for c in w if w[c] > cap + 1e-12]
        if not over:
            break
        excess = sum(w[c] - cap for c in over)
        for c in over:
            w[c] = cap
        free = [c for c in w if w[c] < cap - 1e-12]
        if not free:
            break
        for c in free:
            w[c] += excess / len(free)
    return w


def _cap_group(w: dict[str, float], keymap: dict[str, str], cap: float) -> dict[str, float]:
    w = dict(w)
    for _ in range(50):
        groups: dict[str, float] = {}
        for c, v in w.items():
            g = keymap.get(c, "NA")
            groups[g] = groups.get(g, 0.0) + v
        over = {g: s for g, s in groups.items() if s > cap + 1e-9}
        if not over:
            break
        excess = 0.0
        for g in over:
            members = [c for c in w if keymap.get(c, "NA") == g]
            tot = sum(w[c] for c in members)
            if tot <= 0:
                continue
            f = cap / tot
            for c in members:
                excess += w[c] * (1 - f)
                w[c] *= f
        others = [c for c in w if keymap.get(c, "NA") not in over and w[c] > 0]
        if not others:
            break
        tot_o = sum(w[c] for c in others)
        for c in others:
            w[c] += excess * w[c] / tot_o
    return w


def build_pool(rows: list[dict], cfg: Config,
               tier_map: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """核心-卫星分层 + 目标权重。

    传入 tier_map 时，角色由分级决定：S=核心，A/B=卫星，C 以下不入池。
    不传则退回二道门自身的 core_eligible 判定。
    """
    gate2 = pd.DataFrame(rows)
    if gate2.empty:
        return gate2, pd.DataFrame()

    pool = gate2[gate2["gate2_result"].isin(["pass", "pass_flag", "reactivated"])].copy()
    if tier_map:
        pool["tier"] = pool["fund_code"].map(
            lambda c: (tier_map.get(c) or {}).get("tier"))
        pool = pool[pool["tier"].isin(["S", "A", "B"])].copy()
        # 角色映射：S/A 且满足核心硬条件（风格稳定 + 经理年限 + 规模）才可当核心，
        # 其余 A/B 做卫星。B 不因“规模够大”自动升为核心，避免平庸产品占核心仓。
        core_cond = (pool["drift_level"].eq("stable")
                     & (pool["manager_tenure_years"] >= cfg.core_tenure)
                     & (pool["aum_yi"] > cfg.core_aum_yi))
        pool["core_eligible"] = pool["tier"].isin(["S", "A"]) & core_cond
    watch = gate2[gate2["gate2_result"] == "watchlist"].copy()
    if not watch.empty:
        today = pd.Timestamp.today().normalize()
        watch["enter_date"] = today.strftime("%Y-%m-%d")
        watch["review_deadline"] = (today + pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        watch["status"] = "watching"

    if pool.empty:
        return pool, watch

    core_codes = pool.loc[pool["core_eligible"], "fund_code"].tolist()
    sat_codes = pool.loc[~pool["core_eligible"], "fund_code"].tolist()

    # 风格象限 / 基金公司集中度上限（仅在信息足够时施加上限）
    quad_map = {c: q for c, q in zip(pool["fund_code"], pool["style_quadrant"])
                if isinstance(q, str) and q not in ("unknown", "nan")}
    comp_map = {c: (v if isinstance(v, str) and v else "未知")
                for c, v in zip(pool["fund_code"], pool["company"])}

    w: dict[str, float] = {}
    if core_codes and sat_codes:
        # 核心基金所属公司不再配卫星，避免同一家公司双重暴露把 30% 上限吃满
        core_comps = {comp_map.get(c, "NA") for c in core_codes}
        sat_codes = [c for c in sat_codes if comp_map.get(c, "NA") not in core_comps]
        if not sat_codes:
            w.update(_spread(core_codes, 1.0, cfg.company_cap))
        else:
            # 核心仓受“单一基金公司 30%”约束：两家公司最多 60%，
            # 不做超配再回压，否则会在上限附近来回震荡
            n_comp = len(core_comps)
            core_alloc = min(cfg.core_ratio, cfg.company_cap * n_comp)
            sat_alloc = 1.0 - core_alloc
            sw = _spread(sat_codes, sat_alloc, cfg.satellite_cap)
            leftover = max(sat_alloc - sum(sw.values()), 0.0)
            w.update(_spread(core_codes, core_alloc + leftover, cfg.company_cap))
            w.update(sw)
    elif core_codes:
        w.update(_spread(core_codes, 1.0, cfg.company_cap))
    else:
        # 没有核心仓时，卫星 10% 上限会把 80% 留成现金；此时放宽为等权满仓
        n = len(sat_codes)
        cap = max(cfg.satellite_cap, 1.0 / n) if n else cfg.satellite_cap
        w.update(_spread(sat_codes, 1.0, cap))

    if len(w) > 1:
        # 迭代投影：超限组削到上限，削下的权重分配给“仍有空间”的基金，
        # 直到所有组都不超限；不再做全局归一化，避免把刚压下去的组重新推高
        sat_set = set(sat_codes)
        for _ in range(30):
            for keymap, cap in ((comp_map, cfg.company_cap),
                                (quad_map, cfg.quadrant_cap)):
                if keymap is quad_map and len(set(quad_map.values())) < 2:
                    continue
                sums: dict[str, float] = {}
                for c, v in w.items():
                    g = keymap.get(c, "NA")
                    sums[g] = sums.get(g, 0.0) + v
                for g, s in sums.items():
                    if s <= cap + 1e-9:
                        continue
                    over = [c for c in w if keymap.get(c, "NA") == g]
                    # 先削卫星、再削核心：核心仓是有意配置的底仓，不应被卫星挤占
                    excess = s - cap
                    for group in ([c for c in over if c in sat_set],
                                  [c for c in over if c not in sat_set]):
                        if excess <= 1e-12:
                            break
                        tot_g = sum(w[c] for c in group)
                        if tot_g <= 0:
                            continue
                        take = min(excess, tot_g)
                        for c in group:
                            w[c] -= take * (w[c] / tot_g)
                        excess -= take
            gap = 1.0 - sum(w.values())
            if gap <= 1e-9:
                break
            comp_sum: dict[str, float] = {}
            quad_sum: dict[str, float] = {}
            for c, v in w.items():
                comp_sum[comp_map.get(c, "NA")] = comp_sum.get(comp_map.get(c, "NA"), 0) + v
                quad_sum[quad_map.get(c, "NA")] = quad_sum.get(quad_map.get(c, "NA"), 0) + v
            free = [c for c in w
                    if w[c] < (cfg.satellite_cap if c in sat_set else 1.0) - 1e-9
                    and comp_sum.get(comp_map.get(c), 0) < cfg.company_cap - 1e-9
                    and quad_sum.get(quad_map.get(c, "NA"), 0) < cfg.quadrant_cap - 1e-9]
            if not free:
                break
            base = sum(w[c] for c in free)
            for c in free:
                w[c] += gap * (w[c] / base if base else 1 / len(free))

    core_set = set(core_codes)
    pool["layer"] = ["core" if c in core_set else "satellite" for c in pool["fund_code"]]
    target = {c: round(w.get(c, 0.0), 6) for c in pool["fund_code"]}
    if target:
        # 四舍五入的余数交给卫星仓吸收，避免核心仓被凑整拉低到 60% 以下
        resid = 1.0 - sum(target.values())
        if abs(resid) > 1e-9:
            holders = [c for c in target if c not in core_set] or list(target)
            base = sum(target[c] for c in holders)
            for c in holders:
                target[c] = round(
                    target[c] + resid * (target[c] / base if base else 1 / len(holders)), 6)
    pool["target_weight"] = [target[c] for c in pool["fund_code"]]
    pool = pool[pool["target_weight"] > 0].reset_index(drop=True)
    return pool, watch


def build_rebalance(pool: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """再平衡动作：5pp 偏离触发 + 卫星盈利 20% 落袋。"""
    if pool.empty:
        return pd.DataFrame()

    target = dict(zip(pool["fund_code"], pool["target_weight"]))
    layer = dict(zip(pool["fund_code"], pool["layer"]))
    name = dict(zip(pool["fund_code"], pool["fund_name"]))

    current: dict[str, float] = {}
    pnl: dict[str, float] = {}
    if cfg.holdings and os.path.exists(cfg.holdings):
        h = pd.read_csv(cfg.holdings)
        h.columns = [str(c).strip() for c in h.columns]
        for rec in h.to_dict("records"):
            code = str(rec.get("fund_code")).zfill(6)
            try:
                current[code] = float(rec.get("current_weight", 0) or 0)
            except (TypeError, ValueError):
                current[code] = 0.0
            if "pnl_pct" in rec and pd.notna(rec["pnl_pct"]):
                pnl[code] = float(rec["pnl_pct"])

    actions = []
    initial = not current
    for code in sorted(set(target) | set(current)):
        t = target.get(code, 0.0)
        c = current.get(code, 0.0)
        action, delta, reason = "hold", 0.0, ""

        if initial:
            if t > 0:
                action, delta, reason = "buy", t, "initial_build"
        elif code not in target and c > 0:
            action, delta, reason = "sell", -c, "not_in_pool"
        elif t - c > cfg.rebalance_margin:
            action, delta, reason = "buy", t - c, "weight_deviation"
        elif t - c < -cfg.rebalance_margin:
            action, delta, reason = "sell", t - c, "weight_deviation"

        # 卫星仓盈利落袋：卖出当前仓位的 1/3
        if (not initial) and layer.get(code) == "satellite" and c > 0 \
                and pnl.get(code, 0.0) >= cfg.profit_take:
            take = c / 3
            if action != "sell" or abs(delta) < take:
                action, delta, reason = "sell", -take, "profit_take"

        if action == "hold":
            continue
        actions.append({
            "fund_code": code,
            "fund_name": name.get(code, ""),
            "layer": layer.get(code, ""),
            "action": action,
            "current_weight": round(c, 4),
            "target_weight": round(t, 4),
            "weight_delta": round(delta, 4),
            "amount": round(delta * cfg.total_value, 2) if cfg.total_value > 0 else np.nan,
            "reason": reason,
        })
    return pd.DataFrame(actions)


# ---------------------------------------------------------------- 观察池状态机

DATE_FMT = "%Y-%m-%d"


def _num(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _state_path(cfg: Config) -> str:
    return os.path.join(cfg.state_dir, "watchlist.json")


def load_watch_state(cfg: Config) -> dict:
    path = _state_path(cfg)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_watch_state(state: dict, cfg: Config) -> None:
    os.makedirs(cfg.state_dir, exist_ok=True)
    with open(_state_path(cfg), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _baseline_path(cfg: Config) -> str:
    return os.path.join(cfg.state_dir, "baselines.json")


def load_baselines(cfg: Config) -> dict:
    path = _baseline_path(cfg)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_baselines(baselines: dict, cfg: Config) -> None:
    os.makedirs(cfg.state_dir, exist_ok=True)
    with open(_baseline_path(cfg), "w", encoding="utf-8") as f:
        json.dump(baselines, f, ensure_ascii=False, indent=2)


def update_baselines(rows: list[dict], baselines: dict, today: pd.Timestamp) -> dict:
    """池内基金的滚动基线：用于日后判断“Calmar 是否回升到入库值 80%”。"""
    for row in rows:
        if row.get("gate2_result") in ("pass", "pass_flag", "reactivated"):
            baselines[row["fund_code"]] = {
                "date": today.strftime(DATE_FMT),
                "calmar": _num(row.get("calmar")),
                "quadrant": row.get("style_quadrant"),
                "manager_changes": _num(row.get("manager_changes_3y")),
                "manager_tenure": _num(row.get("manager_tenure_years")),
            }
    return baselines


def _can_revive(row: dict, rec: dict, cfg: Config) -> tuple[bool, list[str]]:
    """复活条件：触发原因消失 + Calmar 回升 + 规模企稳。"""
    miss: list[str] = []
    trigger = f"{rec.get('trigger_reason', '')},{rec.get('trigger_flags', '')}"

    if "significant_drift" in trigger:
        if row.get("style_quadrant") != rec.get("entry_quadrant"):
            miss.append("quadrant_not_back")
    if "manager_changes" in trigger:
        tenure = row.get("manager_tenure_years")
        changes = row.get("manager_changes_3y")
        base = rec.get("entry_manager_changes")
        ok = (pd.notna(tenure) and tenure >= cfg.revive_tenure
              and (base is None or (pd.notna(changes) and changes <= base)))
        if not ok:
            miss.append("manager_unstable")

    base_calmar = rec.get("entry_calmar")
    cur_calmar = row.get("calmar")
    if rec.get("entry_calmar_source") == "pool_baseline" and base_calmar is not None \
            and pd.notna(cur_calmar):
        if cur_calmar < cfg.revive_calmar_ratio * base_calmar:
            miss.append("calmar_low")

    chg = row.get("aum_chg")
    if pd.notna(chg) and abs(chg) > cfg.revive_scale_band:
        miss.append("scale_unstable")
    return (not miss), miss


def update_watchlist(rows: list[dict], state: dict, baselines: dict, cfg: Config,
                     today: pd.Timestamp) -> tuple[dict, list[dict]]:
    """跨期状态机：进入 -> 定期复核 -> 复活 / 退役（6 个月锁）。"""
    events: list[dict] = []
    for row in rows:
        code = row["fund_code"]
        name = row.get("fund_name", "")
        rec = state.get(code)
        status = rec.get("status") if rec else None

        # 1) 退役锁定期内：不参与池，也不重新提交
        if status == "retired":
            lock = pd.Timestamp(rec.get("lock_until", "1970-01-01"))
            if today < lock:
                row["gate2_result"] = "retired_locked"
                row["watch_status"] = "retired"
                continue
            rec["status"] = "expired"
            rec["expired_date"] = today.strftime(DATE_FMT)
            events.append({"fund_code": code, "fund_name": name, "event": "lock_expired",
                           "detail": f"lock_until={rec.get('lock_until')}"})
            status = "expired"

        # 2) 观察中：复活 / 继续观察 / 退役
        if status == "watching":
            if row["gate2_result"] != "watchlist":
                rec.update(status="reactivated", reactivated_date=today.strftime(DATE_FMT),
                           last_review=today.strftime(DATE_FMT), last_check="trigger_resolved")
                row["gate2_result"] = "reactivated"
                row["core_eligible"] = False
                row["watch_status"] = "reactivated"
                events.append({"fund_code": code, "fund_name": name,
                               "event": "reactivated", "detail": "trigger_resolved"})
                continue

            due = (cfg.force_review
                   or (today - pd.Timestamp(rec["last_review"])).days >= cfg.review_interval_days)
            if not due:
                row["watch_status"] = "watching"
                continue

            ok, miss = _can_revive(row, rec, cfg)
            rec["periods"] = int(rec.get("periods", 1)) + 1
            rec["last_review"] = today.strftime(DATE_FMT)
            rec["last_check"] = ",".join(miss) if miss else "resolved"
            if ok:
                rec.update(status="reactivated", reactivated_date=today.strftime(DATE_FMT))
                row["gate2_result"] = "reactivated"
                row["core_eligible"] = False
                row["watch_status"] = "reactivated"
                events.append({"fund_code": code, "fund_name": name,
                               "event": "reactivated", "detail": "conditions_met"})
            elif rec["periods"] >= cfg.max_watch_periods:
                lock_until = today + pd.DateOffset(months=cfg.retire_lock_months)
                rec.update(status="retired", retired_date=today.strftime(DATE_FMT),
                           lock_until=lock_until.strftime(DATE_FMT))
                row["gate2_result"] = "retired_locked"
                row["watch_status"] = "retired"
                events.append({"fund_code": code, "fund_name": name, "event": "retired",
                               "detail": f"periods={rec['periods']},miss={rec['last_check']}"})
            else:
                row["watch_status"] = "watching"
                events.append({"fund_code": code, "fund_name": name, "event": "review_keep",
                               "detail": rec["last_check"]})
            continue

        # 3) 新进入观察池
        if row["gate2_result"] == "watchlist":
            base = baselines.get(code, {})
            base_calmar = base.get("calmar")
            state[code] = {
                "fund_code": code,
                "fund_name": name,
                "status": "watching",
                "trigger_reason": row.get("watch_reason") or "gate2",
                "trigger_flags": row.get("gate2_flags", ""),
                "enter_date": today.strftime(DATE_FMT),
                "review_deadline": (today + pd.DateOffset(months=6)).strftime(DATE_FMT),
                "periods": 0,
                "last_review": today.strftime(DATE_FMT),
                "entry_calmar": base_calmar if base_calmar is not None
                                else _num(row.get("calmar")),
                "entry_calmar_source": "pool_baseline" if base_calmar is not None
                                       else "entry_value",
                "entry_quadrant": (base.get("quadrant")
                                   or row.get("base_quadrant")
                                   or row.get("style_quadrant")),
                "entry_manager_changes": (base.get("manager_changes")
                                          if base.get("manager_changes") is not None
                                          else _num(row.get("manager_changes_3y"))),
                "entry_manager_tenure": _num(row.get("manager_tenure_years")),
                "entry_aum_chg": _num(row.get("aum_chg")),
                "last_check": "entered",
            }
            row["watch_status"] = "watching"
            events.append({"fund_code": code, "fund_name": name, "event": "entered",
                           "detail": state[code]["trigger_reason"]})
        else:
            row["watch_status"] = ""
    return state, events


def watchlist_frame(state: dict, rows: list[dict]) -> pd.DataFrame:
    current = {r["fund_code"]: r for r in rows}
    out = []
    for code, rec in state.items():
        cur = current.get(code, {})
        out.append({
            "fund_code": code,
            "fund_name": rec.get("fund_name") or cur.get("fund_name", ""),
            "status": rec.get("status"),
            "trigger_reason": rec.get("trigger_reason", ""),
            "enter_date": rec.get("enter_date", ""),
            "review_deadline": rec.get("review_deadline", ""),
            "periods": rec.get("periods"),
            "last_review": rec.get("last_review", ""),
            "last_check": rec.get("last_check", ""),
            "reactivated_date": rec.get("reactivated_date", ""),
            "retired_date": rec.get("retired_date", ""),
            "lock_until": rec.get("lock_until", ""),
            "current_quadrant": cur.get("style_quadrant", ""),
            "current_calmar": cur.get("calmar", np.nan),
            "current_aum_chg": cur.get("aum_chg", np.nan),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        order = {"watching": 0, "reactivated": 1, "retired": 2, "expired": 3}
        df["_o"] = df["status"].map(order).fillna(9)
        df = df.sort_values(["_o", "enter_date"]).drop(columns="_o").reset_index(drop=True)
    return df


def _rebalance_state_path(cfg: Config) -> str:
    return os.path.join(cfg.state_dir, "rebalance.json")


def load_rebalance_state(cfg: Config) -> dict:
    path = _rebalance_state_path(cfg)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_rebalance_state(state: dict, cfg: Config) -> None:
    os.makedirs(cfg.state_dir, exist_ok=True)
    with open(_rebalance_state_path(cfg), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def constraint_report(pool: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """第三道门约束校验：分层比例、单只/象限/公司集中度、现金。"""
    if pool.empty:
        return pd.DataFrame([{"item": "可配置池", "limit": "非空", "actual": 0,
                              "passed": False, "detail": "池为空，全部资金留作现金"}])
    total = float(pool["target_weight"].sum())
    core = pool[pool["layer"] == "core"]
    sat = pool[pool["layer"] == "satellite"]
    core_w = float(core["target_weight"].sum())
    sat_w = float(sat["target_weight"].sum())
    quad = pool.groupby("style_quadrant")["target_weight"].sum()
    comp = pool.groupby("company")["target_weight"].sum()

    def row(item, limit, actual, ok, detail=""):
        return {"item": item, "limit": limit, "actual": round(float(actual), 4),
                "passed": bool(ok), "detail": detail}

    rows = [
        row("核心仓位占比", "60%~70%", core_w, 0.60 <= core_w <= 0.70,
            "无核心仓，回退为卫星等权" if core_w == 0 else ""),
        row("卫星仓位占比", "30%~40%", sat_w, 0.30 <= sat_w <= 0.40,
            "核心仓缺失导致卫星超配" if sat_w > 0.40 else "单一卫星上限 10%"),
        row("单一卫星最大权重", "<=10%", float(sat["target_weight"].max()) if len(sat) else 0.0,
            bool(core_w == 0 or len(sat) == 0
                 or sat["target_weight"].max() <= cfg.satellite_cap + 1e-3),
            "无核心仓，豁免单只 10% 上限" if core_w == 0 else ""),
        row("单一风格象限", "<=40%", float(quad.max()),
            bool(quad.max() <= cfg.quadrant_cap + 1e-3)),
        row("单一基金公司", "<=30%", float(comp.max()),
            bool(comp.max() <= cfg.company_cap + 1e-3)),
        row("权重合计", "=100%", total, abs(total - 1) <= 0.005),
        row("现金比例", "<=5%", max(0.0, 1 - total), (1 - total) <= 0.05),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 分级 S~E

TIER_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
TIER_BUDGET_DAYS = {"S": 0, "A": 7, "B": 30, "C": 90, "D": 180, "E": 3650}
# 总分只计“第一道门没有覆盖”的维度，避免收益/回撤/Calmar 被重复奖励
TIER_WEIGHTS = {"stab_power": 0.40, "style_power": 0.25,
                "mgr_power": 0.20, "scale_power": 0.15}
TIER_INFO_KEYS = ["ret_power", "dd_power", "calmar_power"]   # 仅展示，不计分


def _pct100(s: pd.Series, higher_better: bool = True) -> pd.Series:
    r = s.rank(pct=True, na_option="bottom")
    return (r if higher_better else 1 - r) * 100


def _tier_path(cfg: Config) -> str:
    return os.path.join(cfg.state_dir, "tiers.json")


def load_tiers(cfg: Config) -> dict:
    path = _tier_path(cfg)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_tiers(state: dict, cfg: Config) -> None:
    os.makedirs(cfg.state_dir, exist_ok=True)
    with open(_tier_path(cfg), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def bulk_latest_date(cfg: Config) -> pd.Timestamp | None:
    """一次请求拿到全市场最新净值日期，用于变动检测。"""
    cache = os.path.join(cfg.cache_dir, "bulk_nav.csv")
    try:
        if not cfg.refresh and os.path.exists(cache):
            age_h = (time.time() - os.path.getmtime(cache)) / 3600
            if age_h < 12:
                df = pd.read_csv(cache)
            else:
                df = _retry(ak.fund_open_fund_daily_em, retries=cfg.retries, sleep=3.0)
                df.to_csv(cache, index=False, encoding="utf-8-sig")
        else:
            df = _retry(ak.fund_open_fund_daily_em, retries=cfg.retries, sleep=3.0)
            df.to_csv(cache, index=False, encoding="utf-8-sig")
        cols = [c for c in df.columns if str(c).endswith("-单位净值")]
        if not cols:
            return None
        return pd.Timestamp(str(cols[0])[:10])
    except Exception:  # noqa: BLE001
        return None


def tier_scores(df: pd.DataFrame, style_map: dict, gate2_map: dict) -> pd.DataFrame:
    """可靠性打分：只在通过第一道门的样本内做分位。

    收益/回撤/Calmar 已用于准入，这里只保留为展示列，不再计入总分，
    避免同一指标被两道门重复奖励。
    """
    d = df.copy()
    score_keys = list(TIER_WEIGHTS)
    all_keys = score_keys + TIER_INFO_KEYS
    g = d[d["gate1_pass"]].copy()
    if g.empty:
        for k in all_keys:
            d[k] = np.nan
        d["tier_score"] = np.nan
        return d

    g["ret_power"] = _pct100(g["ann_return"])
    g["dd_power"] = _pct100(g["max_drawdown"])
    g["calmar_power"] = _pct100(g["calmar"].fillna(g["calmar"].median()))
    g["stab_power"] = _pct100(g["win_rate"])

    style_r2 = g["fund_code"].map(lambda c: (style_map.get(c) or {}).get("style_r2", np.nan))
    fill = style_r2.median() if style_r2.notna().any() else np.nan
    g["style_power"] = _pct100(style_r2.fillna(fill))
    # 口径统一：二道门判为显著漂移的，风格清晰力直接压到 40 分以下
    drift = g["fund_code"].map(lambda c: (gate2_map.get(c) or {}).get("drift_level"))
    g.loc[drift == "significant_drift", "style_power"] = (
        g.loc[drift == "significant_drift", "style_power"].clip(upper=40))

    tenure = g["fund_code"].map(
        lambda c: (gate2_map.get(c) or {}).get("manager_tenure_years", np.nan))
    g["mgr_power"] = _pct100(tenure.fillna(tenure.median() if tenure.notna().any() else 0))

    # 规模适配力：40 亿附近最优，过小或过大都扣分
    aum = g["fund_code"].map(lambda c: (gate2_map.get(c) or {}).get("aum_yi", np.nan))
    log_dev = np.log(aum.where(aum > 0)) - np.log(40.0)
    fit = 100 * np.exp(-(log_dev ** 2) / (2 * 0.9 ** 2))
    g["scale_power"] = fit.fillna(fit.median() if fit.notna().any() else 50)

    g["tier_score"] = sum(g[k].astype(float) * w for k, w in TIER_WEIGHTS.items()).round(2)
    d = d.merge(g[["fund_code", *all_keys, "tier_score"]], on="fund_code", how="left")
    return d


def assign_tiers(d: pd.DataFrame, gate2_map: dict, prev: dict,
                 today: pd.Timestamp, ref_ret: float = np.nan) -> tuple[pd.DataFrame, list[dict]]:
    dd_cut = d["max_drawdown"].quantile(0.75)
    aum = d["fund_code"].map(lambda c: (gate2_map.get(c) or {}).get("aum_yi", np.nan))
    drift = d["fund_code"].map(lambda c: (gate2_map.get(c) or {}).get("drift_level"))
    status = d["fund_code"].map(lambda c: (gate2_map.get(c) or {}).get("gate2_result"))

    tiers, events = [], []
    for i, row in d.iterrows():
        code = row["fund_code"]
        score = float(row["tier_score"])
        a = aum.get(i)
        g1 = bool(row.get("gate1_pass", False))
        st = status.get(i)
        dr = drift.get(i)

        if pd.notna(a) and a < 0.5:
            t = "E"
        elif not g1 or pd.isna(score):
            t = "D"
        elif (score >= 80 and row["max_drawdown"] >= dd_cut and dr == "stable"
              and st in ("pass", "pass_flag") and row["calmar"] > 0
              # 稳定性不能以长期跑输基准为代价：允许小幅落后，但不能明显掉队
              and (pd.isna(ref_ret) or row["ann_return"] - ref_ret > -0.05)):
            t = "S"
        elif score >= 70 and row["calmar"] > 0:
            t = "A"
        elif score >= 50:
            t = "B"
        elif score >= 30:
            t = "C"
        else:
            t = "D"

        p = prev.get(code) or {}
        pt = p.get("tier")
        pending = p.get("pending")
        force_fr = bool(p.get("force_refresh"))
        event = "hold"
        final = t
        if pt is None:
            event = "new"
        elif t == pt:
            pending = None
        elif TIER_ORDER[t] > TIER_ORDER[pt]:      # 降级立即生效
            event = "downgrade"
            pending = None
        else:                                      # 升级需连续两期确认
            if pending == t:
                event = "upgrade"
                pending = None
            else:
                event = "pending_upgrade"
                pending = t
                final = pt

        tiers.append({"fund_code": code, "company": row.get("company"),
                      "tier": final, "score": round(score, 2), "prev_tier": pt,
                      "event": event,
                      **{k: round(float(row[k]), 1)
                         for k in list(TIER_WEIGHTS) + TIER_INFO_KEYS}})
        if event in ("upgrade", "downgrade", "new"):
            events.append({"date": today.strftime(DATE_FMT), "fund_code": code,
                           "from_tier": pt, "to_tier": final, "event": event,
                           "score": round(score, 2)})
        prev[code] = {"tier": final, "pending": pending,
                      # 级别迁移后下一次运行强制刷新一次，避免“降级反而更久不刷”
                      "force_refresh": force_fr or event in ("upgrade", "downgrade"),
                      "updated": today.strftime(DATE_FMT)}
    return pd.DataFrame(tiers), events


def new_money_multiplier(cfg: Config) -> tuple[float, str, float]:
    """新增资金投入系数（估值分位分档）。优先 PE 分位，缺失时用价格分位代理。"""
    today = pd.Timestamp.today().normalize()
    pct, source = np.nan, "unavailable"
    try:
        hist = ak.index_value_hist_funddb(symbol="沪深300", indicator="市盈率")
        hist["日期"] = pd.to_datetime(hist["日期"])
        pe = hist.set_index("日期")["市盈率"].astype(float).sort_index()
        pe = pe[pe.index >= today - pd.DateOffset(years=5)]
        if len(pe) > 100:
            pct, source = float((pe < pe.iloc[-1]).mean()), "pe_percentile"
    except Exception:  # noqa: BLE001
        pass
    if pd.isna(pct):
        px = fetch_index(BENCH_HS300, cfg)
        px = px[px.index >= today - pd.DateOffset(years=5)]
        pct, source = float((px < px.iloc[-1]).mean()), "price_percentile_proxy"

    if pct < 0.20:
        mult = 3.0
    elif pct < 0.40:
        mult = 2.0
    elif pct < 0.60:
        mult = 1.0
    elif pct < 0.80:
        mult = 0.5
    else:
        mult = 0.0
    return mult, source, pct


# ---------------------------------------------------------------- 风格箱

def _nnls(X: np.ndarray, y: np.ndarray, iters: int = 100, l2: float = 1e-8,
          tol: float = 1e-7) -> np.ndarray:
    """非负最小二乘（坐标下降），第一列为截距、不做非负约束。

    四只风格指数高度相关，普通 OLS 会出现大幅负暴露；加非负约束后
    β 可解释为“对某种风格的净暴露”，更接近风格箱的本意。
    """
    n, p = X.shape
    G = X.T @ X
    c = X.T @ y
    beta = np.zeros(p)
    beta[0] = float(y.mean())
    for _ in range(iters):
        delta = 0.0
        for j in range(1, p):
            num = c[j] - G[j] @ beta + G[j, j] * beta[j]
            den = float(G[j, j]) + l2
            new = max(float(num) / den, 0.0) if den else 0.0
            delta = max(delta, abs(new - beta[j]))
            beta[j] = new
        num0 = c[0] - G[0] @ beta + G[0, 0] * beta[0]
        new0 = float(num0) / float(G[0, 0]) if G[0, 0] else 0.0
        delta = max(delta, abs(new0 - beta[0]))
        beta[0] = new0
        if delta < tol:
            break
    return beta


def _quadrant(betas: dict[str, float]) -> tuple[str, float, float]:
    size_score = (betas["small_value"] + betas["small_growth"]
                  - betas["big_value"] - betas["big_growth"])
    style_score = (betas["big_value"] + betas["small_value"]
                   - betas["big_growth"] - betas["small_growth"])
    size = "大盘" if size_score < -0.05 else ("小盘" if size_score > 0.05 else "中盘")
    style = "价值" if style_score > 0.05 else ("成长" if style_score < -0.05 else "平衡")
    return f"{size}{style}", size_score, style_score


def style_box(nav: pd.Series, style_ret: pd.DataFrame,
              cfg: Config) -> tuple[dict | None, pd.DataFrame]:
    r = nav.pct_change(fill_method=None).dropna()
    common = r.index.intersection(style_ret.index)
    if len(common) < cfg.style_window + 10:
        return None, pd.DataFrame()
    rv = r.reindex(common).to_numpy()
    sv = style_ret.reindex(common).to_numpy()

    snapshots = []
    end = len(common) - 1
    for i in range(cfg.style_periods):
        pos = end - i * cfg.style_step
        if pos + 1 < cfg.style_window:
            break
        y = rv[pos + 1 - cfg.style_window: pos + 1]
        x = sv[pos + 1 - cfg.style_window: pos + 1]
        X = np.column_stack([np.ones(len(x)), x])
        coef = _nnls(X, y)
        y_hat = X @ coef
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
        betas = {k: float(coef[j + 1]) for j, k in enumerate(style_ret.columns)}
        quad, size_score, style_score = _quadrant(betas)
        snapshots.append({
            "date": common[pos].strftime("%Y-%m-%d"),
            **{f"beta_{k}": v for k, v in betas.items()},
            "r2": r2,
            "size_score": size_score,
            "style_score": style_score,
            "quadrant": quad,
        })

    if not snapshots:
        return None, pd.DataFrame()
    hist = pd.DataFrame(snapshots).sort_values("date").reset_index(drop=True)
    quads = hist["quadrant"].tolist()
    changes = sum(1 for a, b in zip(quads, quads[1:]) if a != b)

    def _style_of(q: str) -> str:
        return "价值" if "价值" in q else ("成长" if "成长" in q else "平衡")

    cross_jump = any({_style_of(a), _style_of(b)} == {"价值", "成长"}
                     for a, b in zip(quads, quads[1:]))

    if changes >= cfg.drift_significant or cross_jump:
        drift = "significant_drift"
    elif changes == cfg.drift_mild:
        drift = "mild_drift"
    else:
        drift = "stable"

    # 连续两期确认：最新象限与两个快照前不同，才算“确认漂移”，避免单期噪声误杀
    drift_confirmed = bool(len(quads) >= 3 and quads[-1] != quads[-3])
    base_quadrant = quads[-3] if len(quads) >= 3 else quads[0]  # 漂移前的原象限

    latest = hist.iloc[-1]
    r2 = float(latest["r2"])
    if r2 >= 0.85:
        confidence = "clear"
    elif r2 >= 0.70:
        confidence = "moderate"
    else:
        confidence = "unclear"  # R² < 0.7 视为风格模糊/择时型
    summary = {
        "style_quadrant": latest["quadrant"],
        "style_r2": round(r2, 4),
        "style_confidence": confidence,
        "style_beta_big_value": round(float(latest["beta_big_value"]), 4),
        "style_beta_big_growth": round(float(latest["beta_big_growth"]), 4),
        "style_beta_small_value": round(float(latest["beta_small_value"]), 4),
        "style_beta_small_growth": round(float(latest["beta_small_growth"]), 4),
        "style_shift_count": changes,
        "cross_quadrant_jump": cross_jump,
        "drift_level": drift,
        "drift_confirmed": drift_confirmed,
        "base_quadrant": base_quadrant,
        "style_snapshots": len(hist),
    }
    return summary, hist


# ---------------------------------------------------------------- 主流程

def select_universe(cfg: Config) -> pd.DataFrame:
    if cfg.codes:
        all_funds = _retry(ak.fund_name_em, retries=cfg.retries, sleep=2.0)
        return all_funds[all_funds["基金代码"].isin(cfg.codes)].copy()

    all_funds = _retry(ak.fund_name_em, retries=cfg.retries, sleep=2.0)
    types = EQUITY_TYPES + (INDEX_TYPES if cfg.include_index else [])
    uni = all_funds[all_funds["基金类型"].isin(types)].copy()
    # 去掉同一基金的 C/E 类份额与后端收费代码（东财净值接口对后者返回 HTML）
    uni = uni[~uni["基金简称"].astype(str).str.contains(r"C$|E$|D$|I$|Y$|F$|B$|后端", regex=True)]
    return uni.copy()


def main() -> int:
    ap = argparse.ArgumentParser(description="基金三道门 + 风格箱 MVP")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--codes", type=str, default="")
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--start", type=str, default=None, help="起始日 YYYY-MM-DD，优先于 --years")
    ap.add_argument("--mode", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--include-index", action="store_true")
    ap.add_argument("--min-days", type=int, default=500)
    ap.add_argument("--prescreen-aum", type=float, default=2.0,
                    help="规模预筛下限（亿元），0 表示不筛")
    ap.add_argument("--prescreen-years", type=float, default=3.0,
                    help="成立年限预筛下限（年），0 表示不筛")
    ap.add_argument("--no-prescreen", action="store_true",
                    help="关闭规模/年限预筛")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--failure-ttl", type=int, default=30,
                    help="取数失败缓存有效期（天），到期自动重试")
    ap.add_argument("--clear-failures", action="store_true",
                    help="清空失败缓存（全部重试）")
    ap.add_argument("--no-tier-refresh", action="store_true",
                    help="关闭分级刷新与变动检测（全部走缓存）")
    ap.add_argument("--outdir", type=str, default="output")
    ap.add_argument("--cache-dir", type=str, default="cache")
    ap.add_argument("--holdings", type=str, default=None,
                    help="当前持仓 CSV：fund_code,current_weight[,pnl_pct]")
    ap.add_argument("--total-value", type=float, default=0.0,
                    help="组合总市值，>0 时把权重差换算为金额")
    ap.add_argument("--new-money", type=float, default=0.0,
                    help="本期新增资金，按估值分位系数分批投入")
    ap.add_argument("--state-dir", type=str, default="state")
    ap.add_argument("--force-review", action="store_true",
                    help="忽略评估周期，立即复核观察池")
    ap.add_argument("--force-rebalance", action="store_true",
                    help="同日已生成过指令时仍强制重新生成")
    ap.add_argument("--reset-state", action="store_true",
                    help="清空观察池状态后重新开始")
    args = ap.parse_args()

    cfg = Config(
        start=args.start, years=args.years, mode=args.mode, limit=args.limit,
        codes=[c.strip() for c in args.codes.split(",") if c.strip()],
        include_index=args.include_index, min_days=args.min_days,
        prescreen_aum=args.prescreen_aum, prescreen_years=args.prescreen_years,
        no_prescreen=args.no_prescreen,
        refresh=args.refresh, sleep=args.sleep,
        outdir=args.outdir, cache_dir=args.cache_dir,
        holdings=args.holdings, total_value=args.total_value, new_money=args.new_money,
        state_dir=args.state_dir, force_review=args.force_review,
        force_rebalance=args.force_rebalance,
        failure_ttl_days=args.failure_ttl, clear_failures=args.clear_failures,
        no_tier_refresh=args.no_tier_refresh,
    )

    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    if args.reset_state:
        for path in (_state_path(cfg), _rebalance_state_path(cfg),
                     _baseline_path(cfg), _failure_path(cfg)):
            if os.path.exists(path):
                os.remove(path)
        print("已清空观察池与再平衡状态", flush=True)

    nav_store = load_nav_store(cfg)
    print(f"      净值缓存 {len(nav_store)} 只（单文件 nav_store.pkl）", flush=True)

    end = pd.Timestamp.today().normalize()
    start = (pd.Timestamp(cfg.start) if cfg.start
             else end - pd.DateOffset(days=int(365.25 * cfg.years)))
    print(f"[1/6] 区间 {start.date()} ~ {end.date()}，模式 {cfg.mode}", flush=True)

    print("[2/6] 获取基准与风格指数 …", flush=True)
    bench = build_benchmarks(cfg)
    style_ret = pd.DataFrame({
        k: fetch_index(sym, cfg).pct_change(fill_method=None)
        for k, sym in STYLE_INDEXES.items()
    }).dropna()

    print("[3/6] 构建候选池 …", flush=True)
    uni = select_universe(cfg)
    if not cfg.codes and not cfg.no_prescreen:
        uni = prescreen(uni, cfg)
    if cfg.limit > 0:
        uni = uni.head(cfg.limit)
    print(f"      取净值候选 {len(uni)} 只", flush=True)

    today = pd.Timestamp.today().normalize()
    failures_state = load_failures(cfg)
    if cfg.clear_failures:
        failures_state = {}
    all_records = uni.to_dict("records")
    records, skipped = [], 0
    for rec in all_records:
        c = str(rec["基金代码"])
        if (not cfg.refresh) and is_blacklisted(failures_state.get(c), cfg, today):
            skipped += 1
            continue
        records.append(rec)
    if skipped:
        print(f"      失败缓存跳过 {skipped} 只（{cfg.failure_ttl_days} 天内不再重试）", flush=True)

    # 分级刷新：先检测市场最新净值日，再按 S~E 的刷新预算决定是否更新
    tiers_prev = load_tiers(cfg)
    market_date = None if cfg.no_tier_refresh else bulk_latest_date(cfg)
    plan, due_codes = [], set()
    for rec in records:
        c = str(rec["基金代码"])
        series = _NAV_STORE.get(c) if _NAV_STORE else None
        last = series.index[-1] if series is not None and len(series) else None
        tier_state = tiers_prev.get(c) or {}
        tier = tier_state.get("tier", "B")
        budget = TIER_BUDGET_DAYS.get(tier, 30)
        stale = int((today - last).days) if last is not None else 9999
        changed = bool(market_date is not None and last is not None and market_date > last)
        force_fr = bool(tier_state.get("force_refresh"))
        due = last is None or (changed and stale > budget) or force_fr
        if due and force_fr:
            tier_state["force_refresh"] = False
        if due:
            due_codes.add(c)
        plan.append({"fund_code": c, "tier": tier,
                     "last_nav_date": last.strftime(DATE_FMT) if last is not None else None,
                     "stale_days": stale,
                     "market_date": (market_date.strftime(DATE_FMT)
                                     if market_date is not None else None),
                     "changed": changed, "force_refresh": force_fr, "due": due})
    if market_date is not None:
        print(f"      变动检测：市场最新净值日 {market_date.date()}，"
              f"需刷新 {len(due_codes)}/{len(records)} 只", flush=True)

    codes = [str(r["基金代码"]) for r in records]
    print(f"      取净值 {len(codes)} 只（命中缓存直接返回）…", flush=True)

    rows, failures = [], []
    for i, rec in enumerate(records, 1):
        code = str(rec["基金代码"])
        name = rec["基金简称"]
        ftype = rec["基金类型"]
        try:
            nav = fetch_nav(code, cfg, force=code in due_codes)
            if nav is None or len(nav) < cfg.min_days:
                failures.append({"fund_code": code, "fund_name": name,
                                 "reason": f"净值序列不足({0 if nav is None else len(nav)})"})
                continue
            m = compute_metrics(nav, pick_benchmark(ftype, bench), start, cfg.risk_free)
            if not m:
                failures.append({"fund_code": code, "fund_name": name,
                                 "reason": "指标窗口不足"})
                continue
            m.update({"fund_code": code, "fund_name": name,
                      "fund_type": ftype, "company": None})
            rows.append(m)
        except Exception as exc:  # noqa: BLE001
            failures.append({"fund_code": code, "fund_name": name,
                             "reason": f"{type(exc).__name__}: {str(exc)[:60]}"})
        if i % 300 == 0:
            print(f"      净值 {i}/{len(records)}（有效 {len(rows)}）", flush=True)

    # 同步失败缓存：失败的写黑名单，成功的移出
    for f in failures:
        prev = failures_state.get(f["fund_code"], {})
        failures_state[f["fund_code"]] = {
            "reason": f["reason"],
            "last_try": today.strftime(DATE_FMT),
            "attempts": int(prev.get("attempts", 0)) + 1,
        }
    for m in rows:
        failures_state.pop(m["fund_code"], None)
    save_failures(failures_state, cfg)

    if not rows:
        print("没有有效样本，退出。")
        pd.DataFrame(failures, columns=["fund_code", "fund_name", "reason"]).to_csv(
            os.path.join(cfg.outdir, "data_failures.csv"), index=False, encoding="utf-8-sig")
        save_nav_store(cfg)
        return 1

    print("[4/6] 指标与第一道门 …", flush=True)
    t_stage = time.time()
    df = pd.DataFrame(rows)
    df = apply_gate1(df, cfg.mode)
    print(f"      用时 {time.time() - t_stage:.0f}s", flush=True)

    # 风格箱：对通过第一道门的基金做（池为空则跳过）
    print("[5/6] 风格箱 …", flush=True)
    t_stage = time.time()
    style_rows, hist_rows, style_map = [], [], {}
    passed = df[df["gate1_pass"]]
    for rec in passed.to_dict("records"):
        try:
            nav = fetch_nav(rec["fund_code"], cfg)
            summary, hist = style_box(nav, style_ret, cfg)
            if summary:
                style_map[rec["fund_code"]] = summary
                style_rows.append({"fund_code": rec["fund_code"],
                                   "fund_name": rec["fund_name"], **summary})
                hist = hist.copy()
                hist.insert(0, "fund_code", rec["fund_code"])
                hist_rows.append(hist)
        except Exception as exc:  # noqa: BLE001
            failures.append({"fund_code": rec["fund_code"], "fund_name": rec["fund_name"],
                             "reason": f"style: {type(exc).__name__}: {str(exc)[:60]}"})

    # ---------------- 第二道门
    print(f"      风格箱用时 {time.time() - t_stage:.0f}s", flush=True)
    print("[6/6] 第二道门与组合构建 …", flush=True)
    t_stage = time.time()
    mgr_map = load_manager_map(cfg)
    gate2_rows = []
    for gi, rec in enumerate(passed.to_dict("records"), 1):
        code = rec["fund_code"]
        try:
            mgr = manager_metrics(code, mgr_map, cfg)
            scale = scale_metrics(fetch_scale_history(code, cfg))
            snap = fetch_fund_snapshot(code, cfg)
            style = style_map.get(code) or {}
            g2 = evaluate_gate2(style, mgr, scale, snap, cfg)
            gate2_rows.append({
                "fund_code": code,
                "fund_name": rec["fund_name"],
                "company": snap.get("company"),
                "style_quadrant": style.get("style_quadrant", "unknown"),
                "drift_level": style.get("drift_level", "unknown"),
                "style_r2": style.get("style_r2", np.nan),
                "style_confidence": style.get("style_confidence", "unknown"),
                "style_shift_count": style.get("style_shift_count", np.nan),
                "drift_confirmed": style.get("drift_confirmed", False),
                "base_quadrant": style.get("base_quadrant", "unknown"),
                "calmar": rec.get("calmar"),
                "manager_name": mgr["manager_name"],
                "manager_tenure_years": mgr["manager_tenure_years"],
                "manager_changes_3y": mgr["manager_changes_3y"],
                "manager_product_count": mgr["manager_product_count"],
                "manager_data": mgr["manager_data"],
                "aum_yi": g2["aum_yi"],
                "aum_chg": g2["aum_chg"],
                "scale_quarters": scale["scale_quarters"],
                "gate2_result": g2["gate2_result"],
                "gate2_flags": g2["gate2_flags"],
                "watch_reason": g2["watch_reason"],
                "core_eligible": g2["core_eligible"],
                "turnover_flag": g2["turnover_flag"],
            })
        except Exception as exc:  # noqa: BLE001
            failures.append({"fund_code": code, "fund_name": rec["fund_name"],
                             "reason": f"gate2: {type(exc).__name__}: {str(exc)[:60]}"})
        if gi % 10 == 0:
            print(f"      第二道门 {gi}/{len(passed)}", flush=True)
        time.sleep(cfg.sleep)

    state = load_watch_state(cfg)
    baselines = load_baselines(cfg)
    today = pd.Timestamp.today().normalize()
    state, events = update_watchlist(gate2_rows, state, baselines, cfg, today)
    save_watch_state(state, cfg)

    # ---------------- 分级 S~E：门决定资格，图决定顺序与角色
    # 对标物：红利低波 ETF（512890）同期年化，用于 S 级“不显著跑输”判定
    ref_ret = np.nan
    try:
        ref_nav = fetch_nav("512890", cfg)
        ref_nav = ref_nav[ref_nav.index >= start]
        if len(ref_nav) > 60:
            yrs = max((ref_nav.index[-1] - ref_nav.index[0]).days / 365.25, 1e-6)
            ref_ret = float((ref_nav.iloc[-1] / ref_nav.iloc[0]) ** (1 / yrs) - 1)
    except Exception:  # noqa: BLE001
        pass

    gate2_map = {r["fund_code"]: r for r in gate2_rows}
    tier_df = tier_scores(df, style_map, gate2_map)
    tier_out, tier_events = assign_tiers(tier_df, gate2_map, tiers_prev, today, ref_ret)
    save_tiers(tiers_prev, cfg)
    tier_map = {r["fund_code"]: {"tier": r["tier"]} for r in tier_out.to_dict("records")}

    pool, _ = (build_pool(gate2_rows, cfg, tier_map) if gate2_rows
               else (pd.DataFrame(), pd.DataFrame()))
    actions = build_rebalance(pool, cfg)
    rb_state = load_rebalance_state(cfg)
    duplicated = (not actions.empty
                  and rb_state.get("last_date") == today.strftime(DATE_FMT)
                  and not cfg.force_rebalance)
    if duplicated:
        actions = actions.iloc[0:0]
    if not actions.empty:
        save_rebalance_state({"last_date": today.strftime(DATE_FMT),
                              "last_count": int(len(actions)),
                              "total_value": cfg.total_value}, cfg)
    constraints = constraint_report(pool, cfg)
    watch = watchlist_frame(state, gate2_rows)
    events_df = pd.DataFrame(events)
    baselines = update_baselines(gate2_rows, baselines, today)
    save_baselines(baselines, cfg)

    # 新增资金计划（--new-money > 0 时）
    new_money_df = pd.DataFrame()
    if cfg.new_money > 0 and not pool.empty:
        mult, source, pct = new_money_multiplier(cfg)
        base = pool[pool["layer"] == "core"] if (pool["layer"] == "core").any() else pool
        tot_w = base["target_weight"].sum()
        invest = cfg.new_money * mult
        new_money_df = base[["fund_code", "fund_name", "layer", "style_quadrant"]].copy()
        new_money_df["target_weight"] = base["target_weight"]
        new_money_df["amount"] = (base["target_weight"] / tot_w * invest).round(2) if tot_w else 0.0
        new_money_df["valuation_multiplier"] = mult
        new_money_df["valuation_percentile"] = round(pct, 4)
        new_money_df["percentile_source"] = source

    # ---------------- 分级结果后处理（级别在本轮入库前已确定）
    tier_out = tier_out.merge(df[["fund_code", "fund_name"]], on="fund_code", how="left")
    ann_map = df.set_index("fund_code")["ann_return"]
    tier_out["ann_return"] = tier_out["fund_code"].map(ann_map).round(4)
    tier_out["ref_excess_pp"] = ((tier_out["ann_return"] - ref_ret) * 100).round(2)
    tier_out = tier_out[["fund_code", "fund_name", "tier", "score", "prev_tier", "event",
                         "ret_power", "dd_power", "calmar_power", "stab_power",
                         "style_power", "mgr_power", "scale_power",
                         "ann_return", "ref_excess_pp", "company"]]
    tier_out["_o"] = tier_out["tier"].map(TIER_ORDER)
    tier_out = (tier_out.sort_values(["_o", "score"], ascending=[True, False])
                .drop(columns="_o"))
    tier_counts = tier_out["tier"].value_counts().to_dict()

    # ---------------- 输出
    print(f"      第二道门用时 {time.time() - t_stage:.0f}s", flush=True)
    df.to_csv(os.path.join(cfg.outdir, "fund_metrics.csv"),
              index=False, encoding="utf-8-sig")
    df.to_csv(os.path.join(cfg.outdir, "gate1_report.csv"),
              index=False, encoding="utf-8-sig")
    style_df = pd.DataFrame(style_rows)
    style_df.to_csv(os.path.join(cfg.outdir, "style_box.csv"),
                    index=False, encoding="utf-8-sig")
    if hist_rows:
        pd.concat(hist_rows, ignore_index=True).to_csv(
            os.path.join(cfg.outdir, "style_history.csv"),
            index=False, encoding="utf-8-sig")
    gate2_df = pd.DataFrame(gate2_rows)
    gate2_df.to_csv(os.path.join(cfg.outdir, "gate2_report.csv"),
                    index=False, encoding="utf-8-sig")
    pool.to_csv(os.path.join(cfg.outdir, "fund_pool.csv"),
                index=False, encoding="utf-8-sig")
    actions.to_csv(os.path.join(cfg.outdir, "rebalance_actions.csv"),
                   index=False, encoding="utf-8-sig")
    constraints.to_csv(os.path.join(cfg.outdir, "gate3_report.csv"),
                       index=False, encoding="utf-8-sig")
    watch.to_csv(os.path.join(cfg.outdir, "watchlist.csv"),
                 index=False, encoding="utf-8-sig")
    events_df.to_csv(os.path.join(cfg.outdir, "watchlist_events.csv"),
                     index=False, encoding="utf-8-sig")
    if not new_money_df.empty:
        new_money_df.to_csv(os.path.join(cfg.outdir, "new_money_plan.csv"),
                            index=False, encoding="utf-8-sig")
    tier_out.to_csv(os.path.join(cfg.outdir, "tiers.csv"),
                    index=False, encoding="utf-8-sig")
    pd.DataFrame(plan).to_csv(os.path.join(cfg.outdir, "refresh_plan.csv"),
                              index=False, encoding="utf-8-sig")
    if tier_events:
        hist_path = os.path.join(cfg.outdir, "tier_history.csv")
        hist = pd.DataFrame(tier_events)
        if os.path.exists(hist_path):
            hist = pd.concat([pd.read_csv(hist_path), hist], ignore_index=True)
        hist.drop_duplicates(subset=["date", "fund_code", "event"]).to_csv(
            hist_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(failures, columns=["fund_code", "fund_name", "reason"]).to_csv(
        os.path.join(cfg.outdir, "data_failures.csv"), index=False, encoding="utf-8-sig")
    if failures_state:
        pd.DataFrame([{"fund_code": k, "reason": v.get("reason"),
                       "last_try": v.get("last_try"), "attempts": v.get("attempts")}
                      for k, v in failures_state.items()]).to_csv(
            os.path.join(cfg.outdir, "blacklist.csv"), index=False, encoding="utf-8-sig")

    n = len(df)
    pass_rel = int(df["gate1_pass"].sum())
    pass_abs = int(df["pass_absolute"].sum())
    lines = [
        f"区间：{start.date()} ~ {end.date()}",
        f"有效样本：{n}，取数失败：{len(failures)}，失败缓存跳过：{skipped}",
        f"第一道门（{cfg.mode}）通过：{pass_rel}/{n} = {pass_rel / n:.1%}",
        f"同一批样本按绝对阈值通过：{pass_abs}/{n} = {pass_abs / n:.1%}",
        f"分级分布：{ {k: tier_counts.get(k, 0) for k in 'SABCDE'} }，"
        f"待确认升级 {int((tier_out['event'] == 'pending_upgrade').sum())} 只",
        f"刷新计划：需更新 {len(due_codes)}/{len(records)} 只，"
        f"变动检测={'开' if market_date is not None else '关'}",
        "",
        "各项通过率：",
    ]
    for k, (_, th) in ABSOLUTE_RULES.items():
        rel = (df[f"{k}_pct"] >= RELATIVE_RULES[k]).mean()
        ab = (df[k] >= th).mean()
        lines.append(f"  {k:<14} 相对{PCT_RULE_TEXT[k]} 通过 {rel:5.0%}"
                     f" | 绝对阈值 {th} 通过 {ab:5.0%}")
    if not style_df.empty:
        dist = style_df["drift_level"].value_counts().to_dict()
        lines += ["", f"风格箱：{len(style_df)} 只，漂移分布 {dist}",
                  f"  象限分布 {style_df['style_quadrant'].value_counts().to_dict()}",
                  f"  置信度分布 {style_df['style_confidence'].value_counts().to_dict()}"]
    else:
        lines += ["", "风格箱：无基金进入（第一道门为空池或历史不足）"]

    if not gate2_df.empty:
        lines += ["", f"第二道门：{len(gate2_df)} 只，结果 {gate2_df['gate2_result'].value_counts().to_dict()}"]
        if not watch.empty:
            lines.append(f"  观察池状态 {watch['status'].value_counts().to_dict()}")
        if not events_df.empty:
            lines.append(f"  本期状态机事件 {events_df['event'].value_counts().to_dict()}")
        if not pool.empty:
            core_n = int((pool["layer"] == "core").sum())
            core_w = float(pool.loc[pool["layer"] == "core", "target_weight"].sum())
            lines += [f"可配置池：{len(pool)} 只（核心 {core_n} 只 / {core_w:.0%}）",
                      f"  象限分布 {pool['style_quadrant'].value_counts().to_dict()}"]
            if not actions.empty:
                lines.append(f"  再平衡动作 {len(actions)} 笔：{actions['action'].value_counts().to_dict()}")
            fails = constraints[~constraints["passed"]]
            lines.append(f"  第三道门约束：{len(constraints) - len(fails)}/{len(constraints)} 通过"
                         + (f"，未通过 {fails['item'].tolist()}" if not fails.empty else ""))
        else:
            lines.append("可配置池为空：本期仅观测，不产生交易")
    if duplicated:
        lines.append("同日已生成过再平衡指令，本次跳过（--force-rebalance 可强制执行）")
    if not new_money_df.empty:
        lines.append(f"新增资金：{cfg.new_money:.0f} × 系数 "
                     f"{new_money_df['valuation_multiplier'].iloc[0]}（分位 "
                     f"{new_money_df['valuation_percentile'].iloc[0]:.0%}，"
                     f"{new_money_df['percentile_source'].iloc[0]}）")

    lines += ["", f"中位数：回撤 {df['max_drawdown'].median():.1%}，"
                  f"Calmar {df['calmar'].median():.2f}，Sharpe {df['sharpe'].median():.2f}，"
                  f"IR {df['ir'].median():.2f}，胜率 {df['win_rate'].median():.1%}"]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(cfg.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    save_nav_store(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
