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


def idx_txt(name, d):
    """标题用：上证 3577.62 跌0.28%"""
    price = d.get("price")
    chg = d.get("chg")
    if chg is None:
        return f"{name} {price if price is not None else '–'}"
    word = "涨" if chg > 0 else ("跌" if chg < 0 else "平")
    return f"{name} {price if price is not None else '–'} {word}{abs(chg):.2f}%"


def today_str():
    return day_str()


def day_str(offset=0):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone(timedelta(hours=8)))
            + timedelta(days=offset)).strftime("%Y-%m-%d")


def bj_hour():
    from datetime import datetime, timedelta, timezone
    return datetime.now(timezone(timedelta(hours=8))).hour


def mark_notified(day):
    """推送成功后才回写标记，失败时保留 pendingNotifyDate 以便补发重试。"""
    if not day:
        return
    path = os.path.join("state", "intraday.json")
    try:
        with open(path, encoding="utf-8") as f:
            snap = json.load(f)
    except Exception:  # noqa: BLE001
        return
    if snap.get("date") != day:
        return
    snap["notifiedDate"] = day
    snap["pendingNotifyDate"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False)


def read_payload():
    html = open("output/tier_dashboard.html", encoding="utf-8").read()
    return json.JSONDecoder().raw_decode(html[html.index("const D = ") + 10:])[0]


def read_snapshot():
    try:
        with open(os.path.join("state", "intraday.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def main():
    token = os.environ.get("PUSHPLUS_TOKEN", "").strip()
    mode = sys.argv[1] if len(sys.argv) > 1 else "--push"
    if not token:
        print("未配置 PUSHPLUS_TOKEN，跳过推送")
        return 0
    try:
        if mode == "--heartbeat":
            snap = read_snapshot()
            title = "基金看板 · 月度自检"
            content = (f"<div style='font-size:13px'>定时链路存活。<br>"
                       f"最近快照：{snap.get('date', '–')} {snap.get('time', '–')}"
                       f"（{snap.get('time') and '有' or '无'}数据）<br>"
                       f"<a href='{PAGE}'>打开看板</a></div>")
            return send(token, title, content)

        snap_day = ""
        if mode == "--catch-up":
            snap = read_snapshot()
            snap_day = snap.get("date") or ""
            if snap_day and snap.get("notifiedDate") == snap_day:
                print(f"{snap_day} 已推送，补发跳过")
                return 0
            if not snap_day or snap_day < day_str(-1):
                print(f"快照日期 {snap_day or '无'} 过旧，补发跳过")
                return 0
            if not 8 <= bj_hour() < 22:
                print(f"当前 {bj_hour()} 点，非补发时段（08–22），跳过")
                return 0

        data = read_payload()
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
    label = "补发·" if mode == "--catch-up" else ""
    title = (f"{label}基金看板 {m.get('date', '')[5:]} "
             f"{idx_txt('上证', i1)} {idx_txt('沪深300', i2)}")

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

    rc = send(token, title, content)
    if rc == 0:
        mark_notified(snap_day if mode == "--catch-up" else m.get("date"))
    return rc


def send(token, title, content):
    try:
        r = requests.post(URL, json={"token": token, "title": title[:100],
                                     "content": content, "template": "html"},
                          timeout=20)
        print(r.status_code, r.text[:200])
        ok = r.status_code == 200 and '"code":200' in r.text.replace(" ", "")
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        print(f"PushPlus 推送失败：{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
