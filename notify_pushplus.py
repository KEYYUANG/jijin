#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""14:30 收盘前推送：把当日指数、广度、预测与建议摘要发到 PushPlus。

依赖环境变量 PUSHPLUS_TOKEN；未配置时静默跳过，不影响构建。
"""
import json
import os
import sys
from collections import Counter

import requests

URL = "http://www.pushplus.plus/send"
PAGE = "https://keyyuang.github.io/jijin/"


def pct(v):
    if v is None:
        return "–"
    return f"{v:+.2f}%"


def main():
    token = os.environ.get("PUSHPLUS_TOKEN", "").strip()
    if not token:
        print("未配置 PUSHPLUS_TOKEN，跳过推送")
        return 0
    try:
        html = open("output/tier_dashboard.html", encoding="utf-8").read()
        data, _ = json.JSONDecoder().raw_decode(html[html.index("const D = ") + 10:])
    except Exception as exc:  # noqa: BLE001
        print(f"读取看板数据失败：{exc}")
        return 0

    m = data.get("intraday") or {}
    i1 = m.get("index") or {}
    i2 = m.get("index2") or {}
    p = m.get("pred") or {}
    b = m.get("breadth") or {}
    funds = [f for f in data.get("funds", []) if f.get("tier") in ("S", "A", "B", "C")]
    cnt = Counter((f.get("advice") or {}).get("bucket", "–") for f in funds)
    picks = sorted([f for f in funds if f.get("tier") in ("S", "A")],
                   key=lambda f: -(f.get("score") or 0))[:5]

    time_txt = f"{m.get('date', '')} {m.get('time', '')}".strip()
    title = (f"基金看板 {m.get('date', '')} 上证{pct(i1.get('chg'))} "
             f"沪深300{pct(i2.get('chg'))} · {p.get('dir', '')}")

    rows = "".join(
        f"<tr><td style='padding:3px 6px'>{f['tier']}</td>"
        f"<td style='padding:3px 6px'>{f['name']}</td>"
        f"<td style='padding:3px 6px;text-align:right'>{pct(f.get('chg'))}</td>"
        f"<td style='padding:3px 6px'>{((f.get('advice') or {}).get('bucket') or '–')}</td></tr>"
        for f in picks)

    advice_txt = " / ".join(f"{k} {v}" for k, v in cnt.most_common(5))
    reasons = "；".join(p.get("reasons") or [])

    content = f"""
<div style="font-size:13px;line-height:1.7">
  <div><b>快照</b> {time_txt}</div>
  <div><b>上证</b> {i1.get('price', '–')}　{pct(i1.get('chg'))}　　
       <b>沪深300</b> {i2.get('price', '–')}　{pct(i2.get('chg'))}</div>
  <div><b>估值广度</b> 涨 {b.get('up', '–')} / 跌 {b.get('down', '–')}
       （上涨占比 {b.get('upRatio', '–')}%，样本 {b.get('total', '–')}）</div>
  <div><b>方向预测</b> {p.get('dir', '–')}（置信 {p.get('conf', '–')}%）{('：' + reasons) if reasons else ''}</div>
  <div><b>建议分布</b> {advice_txt}</div>
  <hr style="border:none;border-top:1px solid #eee;margin:8px 0">
  <table style="border-collapse:collapse;font-size:12px;width:100%">{rows}</table>
  <div style="margin-top:8px"><a href="{PAGE}">打开看板</a></div>
  <div style="color:#888;font-size:11px;margin-top:6px">规则打分的启发式结果，不构成投资建议。</div>
</div>"""

    try:
        r = requests.post(URL, json={"token": token, "title": title[:100],
                                     "content": content, "template": "html"},
                          timeout=20)
        print(r.status_code, r.text[:200])
    except Exception as exc:  # noqa: BLE001
        print(f"PushPlus 推送失败：{exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
