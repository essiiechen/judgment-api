#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向：把多种信号一次收集齐（缓存到 signals.json），再离线扫描组合方式。
信号 = 全局投票(k=5/10/20/50) + 案由分层投票(k=5/10/20/50) + 案由先验 + 文本特征
组合 = score_d = share_d^alpha * prior_d^(1-alpha)，扫 alpha；也扫 k 与 全局/分层
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from judge import Judge, load
from 实验_方向优化 import FEAT, DIRS, feats

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "signals.json"


def collect():
    recs = load()
    idx = list(range(len(recs)))
    random.Random(42).shuffle(idx)
    test, train = idx[:300], idx[300:]
    pool = [recs[i] for i in train]
    j = Judge(pool)

    prior = defaultdict(Counter)
    by_cause = defaultdict(set)
    for i, r in enumerate(pool):
        if r["cause"]:
            prior[r["cause"]][r["direction"]] += 1
            by_cause[r["cause"]].add(i)
    prior_p = {c: {d: v[d] / sum(v.values()) for d in DIRS} for c, v in prior.items()}
    overall = Counter(r["direction"] for r in pool)
    tot = sum(overall.values())
    prior_all = {d: overall[d] / tot for d in DIRS}

    out = []
    for i in test:
        r = recs[i]
        hits = j.bm.search(r["front"], topk=200)
        ca = Counter()
        for ii, s in hits[:20]:
            if pool[ii]["cause"]:
                ca[pool[ii]["cause"]] += s
        cause_pred = ca.most_common(1)[0][0] if ca else ""
        pidx = by_cause.get(cause_pred)
        rec_i = {"id": r["id"], "gold": r["direction"], "cause_pred": cause_pred,
                 "cause_gold": r["cause"], "prior": prior_p.get(cause_pred, prior_all),
                 "feat": feats(r["front"]), "share": {}, "share2": {}}
        for k in (5, 10, 20, 50):
            v = Counter()
            for ii, s in hits[:k]:
                v[pool[ii]["direction"]] += s
            t = sum(v.values()) or 1
            rec_i["share"][k] = {d: v[d] / t for d in DIRS}
            v2 = Counter()
            for ii, s in hits:
                if pidx and ii in pidx:
                    v2[pool[ii]["direction"]] += s
                if len(v2) and sum(1 for ii, _ in hits if ii in (pidx or set())) >= k:
                    break
            # 更干净的写法：单独取该案由的前 k 个
            hits2 = [(ii, s) for ii, s in hits if (pidx is None or ii in pidx)][:k]
            v2 = Counter()
            for ii, s in hits2:
                v2[pool[ii]["direction"]] += s
            t2 = sum(v2.values()) or 1
            rec_i["share2"][k] = {d: v2[d] / t2 for d in DIRS}
        out.append(rec_i)
    return out


def scan(data):
    n = len(data)
    base = Counter(d["gold"] for d in data)
    print(f"样本 {n}｜{'全选支持':24s} {base['支持']/n*100:5.1f}%   "
          f"{'全选部分支持':24s} {base['部分支持']/n*100:5.1f}%")

    def sc(d, key, k, alpha, prior_on=True):
        sh = d[key][k]
        s = {}
        for dd in DIRS:
            v = sh[dd] if sh[dd] > 0 else 1e-6
            p = d["prior"][dd] if prior_on else 1.0
            s[dd] = (v ** alpha) * (p ** (1 - alpha))
        return max(s, key=s.get)

    print("\n[k=20] 不同 alpha（1=纯投票，0=纯先验）：")
    for key, name in (("share", "全局投票"), ("share2", "案由分层投票")):
        row = []
        for alpha in (1.0, 0.9, 0.75, 0.5, 0.25, 0.0):
            ok = sum(1 for d in data if sc(d, key, 20, alpha) == d["gold"])
            row.append(f"a={alpha:.2f}:{ok/n*100:.1f}%")
        print(f"  {name:12s} " + "  ".join(row))

    print("\n固定 alpha=1（纯投票），扫 k：")
    for key, name in (("share", "全局投票"), ("share2", "案由分层投票")):
        row = []
        for k in (5, 10, 20, 50):
            ok = sum(1 for d in data if sc(d, key, k, 1.0) == d["gold"])
            row.append(f"k={k}:{ok/n*100:.1f}%")
        print(f"  {name:12s} " + "  ".join(row))

    print("\n案由先验单独用（alpha=0）：")
    row = []
    for k in (5, 10, 20, 50):
        ok = sum(1 for d in data if sc(d, "share", k, 0.0) == d["gold"])
        row.append(f"k={k}:{ok/n*100:.1f}%")
    print("  先验          " + "  ".join(row))

    # 最优组合的混淆矩阵
    best = (0, None)
    for key in ("share", "share2"):
        for k in (5, 10, 20, 50):
            for alpha in (1.0, 0.9, 0.75, 0.5, 0.25, 0.0):
                ok = sum(1 for d in data if sc(d, key, k, alpha) == d["gold"])
                if ok > best[0]:
                    best = (ok, (key, k, alpha))
    ok, (key, k, alpha) = best
    print(f"\n最优：{key} k={k} alpha={alpha} -> {ok/n*100:.1f}%")
    conf = Counter((d["gold"], sc(d, key, k, alpha)) for d in data)
    print("混淆（真实→预测）：", {f"{a}->{b}": c for (a, b), c in conf.most_common()})


def main():
    if CACHE.exists():
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"读缓存 {CACHE}（{len(data)} 条）")
    else:
        data = collect()
        CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"已缓存 -> {CACHE}")
    scan(data)


if __name__ == "__main__":
    main()
