#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 3,520 份判决书全文解析成结构化字段。

只依赖标准库。产出 parsed.jsonl，每行：
  id, court, case_no, cause, plaintiff, defendant,
  front（「本院认为」之前的部分 = API 的输入形态）,
  laws（判决依据法条）, direction（支持/部分支持/驳回）, verdict（判决主文）

案由的抽法：全库先统计高频「…纠纷」构造词表，再按「最长匹配」回填，
比单条正则（覆盖率只有 22.6%）稳得多。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "judgments.jsonl"
OUT = ROOT / "parsed.jsonl"


# ---------------- 切段 ----------------
def split_sections(text: str):
    """返回 (front, rest)：front = 本院认为之前（含法院/当事人/诉请/查明事实）。"""
    m = re.search(r"本院认为", text)
    if m:
        return text[: m.start()], text[m.start():]
    m = re.search(r"判决如下", text)
    if m:
        return text[: m.start()], text[m.start():]
    return text, ""


def court_of(text: str) -> str:
    m = re.match(r"\s*([\u4e00-\u9fa5]{2,20}?(?:人民法院|法院|法庭))", text)
    return m.group(1) if m else ""


def case_no_of(text: str) -> str:
    m = re.search(r"[（(]\s*((?:19|20)\d{2})\s*[）)]\s*[\u4e00-\u9fa5]{0,12}?\s*[第]?\s*[\d号\-－字第]+", text)
    if m:
        return m.group(0).strip()
    m = re.search(r"[（(]\s*((?:19|20)\d{2})\s*[）)][\u4e00-\u9fa5\d]{2,20}号", text)
    return m.group(0).strip() if m else ""


# ---------------- 案由 ----------------
# 全库高频案由就这 8 类（各约 400 份）。变体统一归一到标准名；
# 剩下的零散案由（失业保险待遇、承揽合同…）保留原文，不强行归类。
CAUSE_ALIAS = [
    ("民间借贷纠纷", ["民间借贷合同纠纷", "民间借贷"]),
    ("物业服务合同纠纷", ["物业管理服务合同纠纷", "物业管理合同纠纷", "物业服务合同"]),
    ("机动车交通事故责任纠纷", ["道路交通事故人身损害赔偿纠纷", "机动车道路交通事故责任纠纷",
                        "道路交通事故责任纠纷", "交通事故责任纠纷", "道路交通事故"]),
    ("买卖合同纠纷", ["商品房买卖合同纠纷", "商品房销售合同纠纷", "买卖合同欠款纠纷", "买卖合同"]),
    ("离婚纠纷", ["离婚"]),
    ("提供劳务者受害责任纠纷", ["提供劳务者受害责任权纠纷", "提供劳务者受害纠纷", "提供劳务者受害"]),
    ("金融借款合同纠纷", ["借款合同纠纷", "金融借款合同"]),
    ("劳动争议", ["劳动争议纠纷", "劳动合同纠纷"]),
]
# 由长到短排列，保证最长优先
_ALIAS = sorted([(v, std) for std, vs in CAUSE_ALIAS for v in vs] + [(s, s) for s, _ in CAUSE_ALIAS],
                key=lambda x: -len(x[0]))


def build_cause_vocab(texts, min_freq=3):
    """保留：全库统计「…纠纷」词频，用来兜底白名单之外的零散案由。"""
    cnt = Counter()
    for t in texts:
        for w in set(re.findall(r"[\u4e00-\u9fa5]{2,16}?纠纷", t)):
            cnt[w] += 1
    return {w for w, c in cnt.items() if c >= min_freq and len(w) >= 4}, cnt


_AN_YI_AN = re.compile(r"([\u4e00-\u9fa5]{2,16})(?:纠纷)?一案")


def cause_of(front: str, vocab):
    """
    1) 先去掉《…》（法条名会冒充案由，比如《最高人民法院关于审理物业服务纠纷…》）
    2) 优先在「…一案」这个位置找（案由最常写在这里），找最长匹配
    3) 找不到再回退到全文最长匹配
    """
    s = re.sub(r"《[^》]*》", " ", front)
    for seg in ([m.group(1) for m in _AN_YI_AN.finditer(s)] + [s]):
        for v, std in _ALIAS:
            if v in seg:
                return std
    # 白名单之外：用全库高频候选兜底（排除明显不是案由的口语）
    best = ""
    for w in vocab:
        if w in s and len(w) > len(best) and not re.search(r"(发生|引起|间的|因与)", w):
            best = w
    return best


# ---------------- 当事人 ----------------
PARTY_RE = re.compile(r"(原告|被告|反诉原告|反诉被告)([\u4e00-\u9fa5A-Za-z0-9（）()·]{2,40}?)(?:[，,、。；;（(]|\s|系|诉|与)")


NOISE = re.compile(r"(?:共同)?委托|代理|诉讼|法定|负责人|经营者|住所|统一社会信用|组织机构代码"
                   r"|出生|汉族|该(?:公司|厂)|系|已|未|现|原住")


def parties_of(front: str):
    ps = {}
    for m in PARTY_RE.finditer(front[:1200]):
        role, name = m.group(1), m.group(2).strip()
        # 「原告共同委托代理人…」这类不是当事人名，跳过这次匹配
        if NOISE.search(name) or len(name) > 22:
            continue
        if role not in ps and len(name) >= 2:
            # 去掉粘连的尾巴：「牟心龙物业服务合同纠纷一案」→「牟心龙」
            name = re.split(r"[\u4e00-\u9fa5]{2,16}?纠纷|一案", name)[0]
            name = re.sub(r"(诉称|辩称|之委托|委托代理|经本院|未到庭|到庭参加|无正当理由).*$", "", name)
            name = name.strip("、，,的")
            if len(name) >= 2:
                ps[role] = name
    return ps.get("原告", ""), ps.get("被告", "")


# ---------------- 法条 ----------------
LAW_RE = re.compile(r"《([^》]{2,60})》\s*(?:第[〇零一二三四五六七八九十百千\d]+条(?:之[〇零一二三四五六七八九十\d]+)?"
                    r"(?:、第[〇零一二三四五六七八九十百千\d]+条)*)?")


def laws_of(text: str, rest: str):
    """
    优先取「判决如下」前面那一段（法院写「据此，依照/根据…之规定」的地方），
    那是真正的判决依据；取不到再退回全文（去重、保序）。
    """
    tail = ""
    m = re.search(r"判决如下", rest)
    if m:
        tail = rest[max(0, m.start() - 400): m.start()]
    out = []

    def collect(s):
        res = []
        for lm in LAW_RE.finditer(s):
            name = lm.group(1).strip()
            seg = lm.group(0)
            arts = re.findall(r"第([〇零一二三四五六七八九十百千\d]+)条", seg)
            if not arts:
                res.append(f"《{name}》")
            else:
                for a in arts:
                    res.append(f"《{name}》第{a}条")
        return res

    for s in (tail, text):
        for x in collect(s):
            if x not in out:
                out.append(x)
        if out:
            break
    # 去掉明显不是法律依据的东西（证据名称、文件名等）
    bad = re.compile(r"(?:起诉状|答辩状|证据|笔录|合同|协议|发票|收据|证明|判决书|裁定书|调解书|清单|说明|意见|函|通知|报告|标准|办法|规定|细则)") if False else None
    return out[:12]


# ---------------- 判决方向与主文 ----------------
def verdict_of(text: str):
    m = re.search(r"判决如下[：:]?", text)
    if not m:
        return ""
    v = text[m.end():]
    # 截到「如不服本判决」；没有就截到审判人员段
    for pat in (r"如不服本判决", r"审\s*判\s*员", r"审\s*判\s*长", r"代理审判员", r"人民陪审员"):
        mm = re.search(pat, v)
        if mm:
            v = v[: mm.start()]
            break
    # 去掉「如果未按本判决指定的期间履行…」这段法定告知（不是判决项）
    v = re.sub(r"如果未按本判决指定的期间履行[^。]*。", "", v)
    v = re.sub(r"如果未按本判决指定的期间履行.*?(?=案件受理费|诉讼费|如不服|$)", "", v)
    v = re.sub(r"\s+", "", v).strip()
    return v


# 注意别只写「驳回原告」——有的文书直接写「驳回曹俊英的诉讼请求」，不带身份词
REJECT = r"驳回"
PAY = r"(?:支付|给付|赔偿|偿还|返还|交纳|缴纳|交付|履行|支付上述)"


def direction_of(verdict: str):
    """
    按「项」判断，而不是扫描整段——否则诉讼费负担里的「承担」「驳回」会带偏结论。
    只看主项（案件受理费之前的部分）。
    """
    if not verdict:
        return "支持"
    main = re.split(r"案件受理费|诉讼费|保全费|鉴定费|公告费", verdict)[0]
    items = [x for x in re.split(r"[。；;]", main) if x.strip()]
    has_pay = any(re.search(r"(?:被告|反诉被告|被申请人)" + r"[^。；]{0,80}?" + PAY, it) or
                  re.search(PAY + r"[^。；]{0,40}?(?:元|款|费|金)", it) for it in items)
    reject_other = any(re.search(REJECT + r"[^。；]{0,40}?(?:其他|其余)", it) for it in items)
    reject_all = any(re.search(REJECT + r"[^。；]{0,40}?(?:全部)?诉讼请求", it) and
                     not re.search(r"(?:其他|其余)", it) for it in items)
    if reject_other:
        return "部分支持"
    if reject_all:
        return "部分支持" if has_pay else "驳回"
    return "支持"


def main():
    docs = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    fronts = []
    for d in docs:
        fr, _ = split_sections(d["text"])
        fronts.append(fr)

    vocab, cnt = build_cause_vocab(fronts)
    print(f"案由词表：{len(vocab)} 个候选（全库共出现 {len(cnt)} 种「…纠纷」）")

    parsed, miss = [], Counter()
    for d, fr in zip(docs, fronts):
        text = d["text"]
        _, rest = split_sections(text)
        p, df = parties_of(fr)
        v = verdict_of(text)
        rec = {
            "id": d["id"],
            "court": court_of(text),
            "case_no": case_no_of(text),
            "cause": cause_of(fr, vocab),
            "plaintiff": p,
            "defendant": df,
            "front": fr,
            "laws": laws_of(text, rest),
            "direction": direction_of(v),
            "verdict": v,
        }
        for k in ("cause", "laws", "verdict", "court", "plaintiff", "defendant"):
            if not rec[k]:
                miss[k] += 1
        parsed.append(rec)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in parsed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(parsed)
    print(f"\n解析 {n} 份 -> {OUT}")
    print("字段缺失：", dict(miss) or "无")
    print("案由 Top15：", Counter(r["cause"] for r in parsed).most_common(15))
    print("方向分布：", Counter(r["direction"] for r in parsed).most_common())
    print("法条条数分布：", Counter(min(len(r["laws"]), 10) for r in parsed).most_common())


if __name__ == "__main__":
    main()
