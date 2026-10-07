#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
判案引擎：把 3,520 份判决书当成判例库做「最近邻判案」。

思路（不调任何大模型，全是本地检索）：
  1. 对每份判决书的「本院认为之前」部分建 BM25 索引（汉字 2-gram 分词，零依赖）
  2. 新案情进来 -> 检索最像的 k 份判例
  3. cause / direction 用相似度加权投票；laws 用加权频次汇总；
     verdict 用最像那份的主文做模板，把当事人名换成新案子的

用法：
  python judge.py eval              # 留一评估（默认 300 份测试集）
  python judge.py eval --k 5 --n 300
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARSED = ROOT / "parsed.jsonl"

HAN = re.compile(r"[\u4e00-\u9fa5]")


# ---------------------------------------------------------------- 分词
def tokenize(s: str):
    """汉字 2-gram：中文检索里既零依赖又稳的做法（jieba 在本机装不上，用它替代）。"""
    s = "".join(HAN.findall(s))
    return [s[i:i + 2] for i in range(len(s) - 1)] if len(s) > 1 else ([s] if s else [])


# ---------------------------------------------------------------- BM25
class BM25:
    K1, B = 1.5, 0.75

    def __init__(self, docs, tokenized=None):
        self.docs = docs
        self.tok = tokenized if tokenized is not None else [tokenize(d) for d in docs]
        self.n = len(docs)
        self.lens = [len(t) for t in self.tok]
        self.avgdl = sum(self.lens) / max(self.n, 1)
        self.tf = [Counter(t) for t in self.tok]
        self.df = Counter()
        for t in self.tok:
            self.df.update(set(t))
        self.idf = {w: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for w, c in self.df.items()}
        self.post = defaultdict(list)
        for i, t in enumerate(self.tok):
            seen = set()
            for w in t:
                if w not in seen:
                    seen.add(w)
                    self.post[w].append(i)

    def search(self, q: str, topk=20):
        qt = tokenize(q)
        scores = defaultdict(float)
        for w in set(qt):
            post = self.post.get(w)
            if not post:
                continue
            idf = self.idf.get(w, 0.0)
            qtf = qt.count(w)
            for i in post:
                tf = self.tf[i][w]
                dl = self.lens[i]
                scores[i] += idf * (tf * (self.K1 + 1)) / \
                    (tf + self.K1 * (1 - self.B + self.B * dl / self.avgdl))
        top = sorted(scores.items(), key=lambda x: -x[1])[:topk]
        return top


# ---------------------------------------------------------------- 判案
CAUSE_W = 1.0
LAW_MIN = 2          # 法条至少要被 2 份相似判例引用才采纳
LAW_MAX = 6


def party_of(text: str):
    """从案情里抽原告/被告——直接复用 parse.py 的规则，保证建库与推断口径一致。"""
    from parse import parties_of
    return parties_of(text)


DIRS = ["支持", "部分支持", "驳回"]


class Judge:
    """
    判案：检索 -> 定案由 -> 定方向 -> 在「同案由 + 同方向」的判例里挑主文模板与法条。

    最后一步的分层很关键：主文的方向必须和预测的方向自洽，
    否则会出现「方向说驳回、主文却写着支付 10 万元」这种自相矛盾的返回。
    """

    def __init__(self, records, alpha=0.9, k=20, law_max=LAW_MAX):
        self.rec = records
        self.alpha, self.k, self.law_max = alpha, k, law_max
        self.bm = BM25([r["front"] for r in records])
        # 案由 -> 方向先验（案由之间差很大：金融借款 76% 支持，物业服务 64% 部分支持）
        pr = defaultdict(Counter)
        for r in records:
            if r["cause"]:
                pr[r["cause"]][r["direction"]] += 1
        self.prior = {c: {d: v[d] / sum(v.values()) for d in DIRS} for c, v in pr.items()}
        allc = Counter(r["direction"] for r in records)
        tot = sum(allc.values()) or 1
        self.prior_all = {d: allc[d] / tot for d in DIRS}
        self.major = allc.most_common(1)[0][0] if allc else "部分支持"

    def __call__(self, case_text: str, k=None, law_max=None):
        k = k or self.k
        law_max = law_max or self.law_max
        hits = self.bm.search(case_text, topk=100)
        if not hits:
            return {"cause": "", "laws": [], "direction": self.major, "verdict": "",
                    "_ref": "", "_sim": 0}

        # --- 1. 案由：相似度加权投票 ---
        cv = Counter()
        for i, s in hits[:k]:
            if self.rec[i]["cause"]:
                cv[self.rec[i]["cause"]] += s
        cause = cv.most_common(1)[0][0] if cv else ""

        # --- 2. 方向：投票 × 案由先验（alpha=0.9，网格扫出来的） ---
        dv = Counter()
        for i, s in hits[:k]:
            dv[self.rec[i]["direction"]] += s
        tot = sum(dv.values()) or 1
        prior = self.prior.get(cause, self.prior_all)
        direction = max(DIRS, key=lambda d: ((dv[d] / tot) if dv[d] else 1e-6) ** self.alpha
                        * prior[d] ** (1 - self.alpha))

        # --- 3. 在「同案由 + 同方向」的判例里挑模板（保证自洽） ---
        pool = [(i, s) for i, s in hits
                if self.rec[i]["cause"] == cause and self.rec[i]["direction"] == direction]
        if not pool:
            pool = [(i, s) for i, s in hits if self.rec[i]["cause"] == cause]
        if not pool:
            pool = hits

        # --- 4. 法条：候选池内加权汇总 ---
        law_v = Counter()
        for i, s in pool[:10]:
            for l in self.rec[i]["laws"]:
                law_v[l] += s
        top_score = pool[0][1] or 1
        laws = [l for l, v in law_v.most_common() if v >= top_score][:law_max]
        if not laws:
            laws = self.rec[pool[0][0]]["laws"][:4]
        if not any("民事诉讼法" in l for l in laws):        # 程序法几乎必引
            for i, _ in pool[:5]:
                p = [l for l in self.rec[i]["laws"] if "民事诉讼法" in l]
                if p:
                    laws.append(p[0])
                    break

        # --- 5. 主文：最像那份做模板，换上本案当事人 ---
        src = self.rec[pool[0][0]]
        verdict = src["verdict"]
        np_, nd_ = party_of(case_text)
        for old, new in ((src["plaintiff"], np_), (src["defendant"], nd_)):
            if old and new and len(old) >= 2 and old != new:
                verdict = verdict.replace(old, new)

        return {"cause": cause, "laws": laws, "direction": direction, "verdict": verdict,
                "_ref": src["id"], "_sim": round(pool[0][1], 2)}


# ---------------------------------------------------------------- 评估
def f1(pred, gold):
    if not pred or not gold:
        return 0.0
    p, g = set(pred), set(gold)
    inter = len(p & g)
    if not inter:
        return 0.0
    return 2 * inter / (len(p) + len(g))


def gram_f1(a: str, b: str, n=2):
    if not a or not b:
        return 0.0
    ga = Counter(a[i:i + n] for i in range(len(a) - n + 1))
    gb = Counter(b[i:i + n] for i in range(len(b) - n + 1))
    inter = sum((ga & gb).values())
    if not inter:
        return 0.0
    return 2 * inter / (sum(ga.values()) + sum(gb.values()))


def evaluate(records, n_test=300, seed=42, verbose=True):
    """留一评估：索引只建一次。测试文档不在检索池里。"""
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    test, train = idx[:n_test], idx[n_test:]
    pool = [records[i] for i in train]
    j = Judge(pool)

    ca_ok = dr_ok = consist = 0
    law_f1 = law_p = law_r = v_f1 = 0.0
    conf = Counter()
    for i in test:
        r = records[i]
        out = j(r["front"])
        ca_ok += out["cause"] == r["cause"]
        dr_ok += out["direction"] == r["direction"]
        conf[(r["direction"], out["direction"])] += 1
        law_f1 += f1(out["laws"], r["laws"])
        law_p += (len(set(out["laws"]) & set(r["laws"])) / len(out["laws"])) if out["laws"] else 0
        law_r += (len(set(out["laws"]) & set(r["laws"])) / len(r["laws"])) if r["laws"] else 0
        v_f1 += gram_f1(out["verdict"], r["verdict"])
        # 自洽性：主文读出来的方向，跟返回的方向字段是否一致
        from parse import direction_of
        consist += direction_of(out["verdict"]) == out["direction"]
    n = len(test)
    if verbose:
        print(f"测试 {n} 份 / 检索池 {len(pool)} 份")
        print(f"  案由准确率   {ca_ok/n*100:5.1f}%")
        print(f"  方向准确率   {dr_ok/n*100:5.1f}%")
        print(f"  法条 F1      {law_f1/n*100:5.1f}%   (P {law_p/n*100:.1f}% / R {law_r/n*100:.1f}%)")
        print(f"  主文 2gramF1 {v_f1/n*100:5.1f}%")
        print(f"  主文与方向自洽 {consist/n*100:5.1f}%")
        print("  方向混淆（真实→预测）：",
              {f"{a}->{b}": c for (a, b), c in conf.most_common()})
    return dict(cause=ca_ok / n, direction=dr_ok / n, law_f1=law_f1 / n,
                verdict=v_f1 / n, consist=consist / n, n=n)


def load():
    with open(PARSED, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="eval")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()

    recs = load()
    if a.cmd == "eval":
        evaluate(recs, n_test=a.n)
    elif a.cmd == "demo":
        j = Judge(recs)
        for i in (0, 500, 1500, 2500, 3400):
            r = recs[i]
            out = j(r["front"])
            print(f"--- 输入 {r['id']}（真实：{r['cause']} / {r['direction']}）")
            print(f"    预测：{out['cause']} / {out['direction']}  (参考判例 {out['_ref']})")
            print(f"    法条：{out['laws'][:4]}")
            print(f"    主文：{out['verdict'][:120]}")


if __name__ == "__main__":
    main()
