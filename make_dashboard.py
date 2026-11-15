#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成六力图看板：读取 output/ 下的分级与指标结果，输出单文件 HTML。"""
import json
import os
import sys

import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
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


def read(name, **kw):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"fund_code": str}, **kw)


def build_records():
    t = read("tiers.csv")
    if t.empty:
        sys.exit("缺少 output/tiers.csv，请先运行 fund_mvp.py")
    t["fund_code"] = t["fund_code"].str.zfill(6)
    for df in (read("fund_metrics.csv"), read("gate2_report.csv"), read("style_box.csv")):
        if not df.empty:
            df["fund_code"] = df["fund_code"].str.zfill(6)

    m = read("fund_metrics.csv")
    g = read("gate2_report.csv")
    s = read("style_box.csv")
    if not m.empty:
        t = t.merge(m[["fund_code", "max_drawdown", "calmar", "sharpe"]],
                    on="fund_code", how="left")
    if not g.empty:
        t = t.merge(g[["fund_code", "aum_yi", "drift_level"]], on="fund_code", how="left")
    if not s.empty:
        t = t.merge(s[["fund_code", "style_quadrant", "style_r2"]], on="fund_code", how="left")

    def num(v, nd=3):
        try:
            f = float(v)
            return None if pd.isna(f) else round(f, nd)
        except (TypeError, ValueError):
            return None

    recs = []
    for _, r in t.iterrows():
        recs.append({
            "code": r["fund_code"], "name": str(r.get("fund_name") or ""),
            "tier": r["tier"], "score": num(r.get("score"), 1) or 0,
            "powers": [num(r.get(k), 0) or 0 for k in DIM_KEYS],
            "ann": num(r.get("ann_return"), 4), "excess": num(r.get("ref_excess_pp"), 1),
            "mdd": num(r.get("max_drawdown"), 4), "calmar": num(r.get("calmar"), 2),
            "sharpe": num(r.get("sharpe"), 2), "aum": num(r.get("aum_yi"), 1),
            "quad": (str(r["style_quadrant"]) if pd.notna(r.get("style_quadrant")) else ""),
            "drift": (str(r["drift_level"]) if pd.notna(r.get("drift_level")) else ""),
            "company": str(r.get("company") or ""),
        })
    recs.sort(key=lambda x: (TIERS.index(x["tier"]), -x["score"]))
    return recs


def main():
    recs = build_records()
    counts = {k: sum(1 for r in recs if r["tier"] == k) for k in TIERS}
    gaps = [r["ann"] - r["excess"] / 100 for r in recs
            if r["ann"] is not None and r["excess"] is not None]
    ref = round(sum(gaps) / len(gaps), 4) if gaps else 0.0
    payload = {"funds": recs, "counts": counts, "tiers": TIERS, "ref": ref,
               "tierDesc": TIER_DESC, "dims": DIM_NAMES}

    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "dashboard_template.html"), encoding="utf-8").read()
    html = tpl.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    dst = os.path.join(OUT, "tier_dashboard.html")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成 {dst}（{len(recs)} 只基金，{os.path.getsize(dst) / 1024:.0f} KB）")
    print("分级分布：" + "  ".join(f"{k}={counts[k]}" for k in TIERS if counts[k]))


if __name__ == "__main__":
    main()
