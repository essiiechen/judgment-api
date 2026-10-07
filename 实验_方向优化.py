#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向的专项优化实验：索引只建一次，快速比较几种判方向的做法。

基线：全局 top-k 加权投票 ≈ 59%。想试的：
  A 基线（全局投票）
  B 案由分层：先定案由，只在该案由的判例里投票
  C 案由先验 × 投票分数（贝叶斯式相乘）
  D 文本特征（被告抗辩强度 / 证据表述）+ 投票，用逻辑回归融合
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from judge import BM25, Judge, load, tokenize

ROOT = Path(__file__).resolve().parent
DIRS = ["支持", "部分支持", "驳回"]

# ---- 文本特征：案情里常见的「抗辩 / 举证不足 / 自认」表述 ----
FEAT = {
    "has_defense": r"被告辩称|被告答辩|辩称",
    "def_deny": r"不认可|不予认可|不承认|否认|没有事实|不同意|不属实",
    "no_evidence": r"未能提供证据|没有证据|未提供证据|举证不能|不足以证明|缺乏证据|没有提交证据",
    "admit": r"被告认可|被告承认|对原告主张.{0,10}无异议|予以认可",
    "part_pay": r"已支付|已偿还|已归还|尚欠|剩余|部分支付",
    "counterclaim": r"反诉",
    "settled": r"调解|和解",
    "absent": r"未到庭|缺席|经本院合法传唤",
}


def feats(text: str):
    return [1.0 if re.search(p, text) else 0.0 for p in FEAT.values()]


class LR:
    """三分类 softmax 回归，纯 Python（数据量小，够用）。"""

    def __init__(self, n_in, n_out=3, lr=0.08, epochs=120):
        self.w = [[0.0] * n_in for _ in range(n_out)]
        self.b = [0.0] * n_out
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        n = len(X)
        for ep in range(self.epochs):
            gw = [[0.0] * len(X[0]) for _ in range(3)]
            gb = [0.0] * 3
            for xi, yi in zip(X, y):
                p = self.prob(xi)
                for c in range(3):
                    d = p[c] - (1.0 if c == yi else 0.0)
                    for j, v in enumerate(xi):
                        gw[c][j] += d * v
                    gb[c] += d
            for c in range(3):
                for j in range(len(X[0])):
                    self.w[c][j] -= self.lr * gw[c][j] / n
                self.b[c] -= self.lr * gb[c] / n
        return self

    def prob(self, x):
        z = [sum(w * v for w, v in zip(self.w[c], x)) + self.b[c] for c in range(3)]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def predict(self, x):
        return max(range(3), key=lambda c: self.prob(x)[c])


def build(recs):
    j = Judge(recs)
    return j


def vote_scores(j, q, k, pool_idx=None):
    """返回三个方向的加权票数（可限定候选池）。"""
    hits = j.bm.search(q, topk=200)
    if pool_idx is not None:
        hits = [(i, s) for i, s in hits if i in pool_idx]
    top = hits[:k]
    v = Counter()
    for i, s in top:
        v[j.rec[i]["direction"]] += s
    tot = sum(v.values()) or 1.0
    return {d: v[d] / tot for d in DIRS}, (len(top) and top)


def main():
    recs = load()
    idx = list(range(len(recs)))
    random.Random(42).shuffle(idx)
    test, train = idx[:300], idx[300:]
    pool = [recs[i] for i in train]
    j = build(pool)

    # 案由 -> 方向先验（只在训练集上算，避免泄漏）
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

    print("案由先验（训练集）：")
    for c, p in sorted(prior_p.items(), key=lambda x: -sum(prior[x[0]].values()))[:9]:
        n = sum(prior[c].values())
        print(f"  {c:16s} n={n:4d}  支持{p['支持']:.2f} 部分{p['部分支持']:.2f} 驳回{p['驳回']:.2f}")

    # ---- 收集测试样本的特征与标签 ----
    Xs, ys, meta = [], [], []
    for i in test:
        r = recs[i]
        share, top = vote_scores(j, r["front"], 20)
        ca = Counter()
        for ii, s in (top or [])[:20]:
            if pool[ii]["cause"]:
                ca[pool[ii]["cause"]] += s
        cause_pred = ca.most_common(1)[0][0] if ca else ""
        pr = prior_p.get(cause_pred, prior_all)
        # 案由分层投票
        pidx = by_cause.get(cause_pred)
        share2, _ = vote_scores(j, r["front"], 20, pidx)
        Xs.append([share[d] for d in DIRS] + [share2[d] for d in DIRS] +
                  [pr[d] for d in DIRS] + feats(r["front"]))
        ys.append(DIRS.index(r["direction"]))
        meta.append((r, share, share2, pr, cause_pred))

    # ---- 逐个策略评估 ----
    def acc(fn, name):
        ok = sum(1 for m, y in zip(meta, ys) if DIRS.index(fn(m)) == y)
        print(f"  {name:28s} {ok/len(meta)*100:5.1f}%")
        return ok / len(meta)

    print("\n方向准确率（测试 300 份）：")
    base = Counter(r["direction"] for r, *_ in meta)
    print(f"  {'全选支持（基线）':28s} {base['支持']/len(meta)*100:5.1f}%")
    acc(lambda m: max(m[1], key=m[1].get), "A 全局top20投票")
    acc(lambda m: max(m[2], key=m[2].get), "B 案由分层top20投票")
    acc(lambda m: max({d: m[1][d] * m[3][d] for d in DIRS},
                      key=lambda d: m[1][d] * m[3][d]), "C 全局投票×案由先验")
    acc(lambda m: max({d: m[2][d] * m[3][d] for d in DIRS},
                      key=lambda d: m[2][d] * m[3][d]), "C2 分层投票×案由先验")

    # ---- D：逻辑回归融合（5 折，防止自欺） ----
    n = len(Xs)
    folds = [(Xs[:240], ys[:240], Xs[240:], ys[240:])]  # 前 240 训练，后 60 测
    print("\nD 逻辑回归（240 训练 / 60 测试，特征=全局投票+分层投票+先验+文本）：")
    for Xtr, ytr, Xte, yte in folds:
        lr = LR(len(Xtr[0])).fit(Xtr, ytr)
        ok = sum(1 for x, y in zip(Xte, yte) if lr.predict(x) == y)
        print(f"  融合模型 {ok/len(yte)*100:5.1f}%")
        print("  特征权重（支持类前 5）：",
              sorted(zip([f"share_{d}" for d in DIRS] + [f"share2_{d}" for d in DIRS] +
                         [f"prior_{d}" for d in DIRS] + list(FEAT.keys()), lr.w[0]),
                     key=lambda x: -abs(x[1]))[:6])


if __name__ == "__main__":
    main()
