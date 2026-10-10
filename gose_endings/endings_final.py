#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""語尾のまとめ方 final の確定と最終数値(第三弾 企画判断用)

v1    = part1_4_endings.py / bts_compare.py と同じ(ㅂ니다→습니다、을까→ㄹ까、요の有無は同じ項目)
final = v1 + 으・은・을・읍・는 の有無の統合(規則表に書いたものだけ) + 에요/예요・単独の 요 の扱い
前処理V2・行末の決め方は part1_4_endings.py と同じ。既存の出力は上書きしない(既定 out_endings_final)。
kpop_analyze.py は読むだけで実行しない。

使い方(コーパス分析フォルダで):
  python3 endings_final.py --gose seventeen_analysis/output/ani_textbook_pipeline/corpus.csv \
     --bts bts_pipeline/bts_textbook_pipeline/bts_corpus.csv --out out_endings_final
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

import kiwipiepy
from kiwipiepy import Kiwi

BR = re.compile(r"\[[^\]]*\]")
PA = re.compile(r"\([^)]*\)")
HANGUL = re.compile(r"[가-힣]")
JAMO = str.maketrans({"ᆫ": "ㄴ", "ᆯ": "ㄹ", "ᆸ": "ㅂ", "ᆷ": "ㅁ", "ᆻ": "ㅆ"})
NO_STRIP = {"세요", "에요", "예요"}
V1_EF = {"ㅂ니다": "습니다", "을까": "ㄹ까"}
F_EF = {"을까": "ㄹ까", "을게": "ㄹ게", "을래": "ㄹ래", "으세요": "세요", "으세": "세요", "은가": "ㄴ가", "는가": "ㄴ가",
        "은데": "ㄴ데", "는데": "ㄴ데", "는다": "ㄴ다", "으냐": "냐", "느냐": "냐", "ㅂ니다": "습니다", "ㅂ니까": "습니까",
        "읍시다": "ㅂ시다", "으니": "니", "으라": "라"}
F_EC = {"으면": "면", "은데": "ㄴ데", "는데": "ㄴ데", "으니까": "니까", "으러": "러", "으려고": "려고", "을게": "ㄹ게"}
F_EP = {"으시": "시"}
DISPLAY = {"ㄹ까": "-(으)ㄹ까", "ㄹ게": "-(으)ㄹ게", "ㄹ래": "-(으)ㄹ래", "세요": "-(으)세요", "ㄴ가": "-(으/느)ㄴ가",
           "ㄴ데": "-(으/느)ㄴ데", "ㄴ다": "-(느)ㄴ다", "냐": "-(으/느)냐", "습니다": "-(스)ㅂ니다", "습니까": "-(스)ㅂ니까",
           "ㅂ시다": "-(으)ㅂ시다", "니": "-(으)니", "라": "-(으)라", "면": "-(으)면", "니까": "-(으)니까", "러": "-(으)러",
           "려고": "-(으)려고", "시": "-(으)시"}
GOSE14_V1 = ["어", "야", "지", "습니다", "다", "네", "잖아", "세요", "자", "ㄹ게", "ㄹ까", "ㄴ다", "ㄴ가", "거든"]
NOUNISH = ("NNG", "NNP", "NNB", "NR", "NP", "MAG", "MAJ")
PRED = ("VV", "VA", "VX", "VCP", "VCN", "XSV", "XSA", "EP")
CATS = ("EF", "EC", "その他", "語尾なし", "未区分")
P0 = {"GOSE": {"全行": 102006, "EF": 50783, "EC": 11090, "その他": 891, "語尾なし": 39242, "14種類": 44020},
      "run": {"全行": 120127, "EF": 57388, "14種類": 47396},
      "bomb": {"全行": 35907, "EF": 16684, "14種類": 14083}}
PART4_WORDS = ["그래", "그래요", "그러니까", "그럼", "어떡해", "왜"]
DL_FORMS = ["들어", "들었", "걸어", "걸었", "물어", "물었"]
SEED = 42


def clean(t):
    return re.sub(r"\s+", " ", PA.sub(" ", BR.sub(" ", t))).strip()


def nf(f):
    return f.translate(JAMO)


def strip_yo(form):
    form = nf(form)
    if form == "죠":
        return "지", True
    if form in NO_STRIP:
        return form, True
    if form.endswith("요") and len(form) > 1:
        return form[:-1], True
    return form, False


def bt(tag):
    return tag.split("-")[0]


def an(core):
    return " ".join(f"{t.form}/{t.tag}" for t in core)


def pct(a, b, d=1):
    return f"{a / b * 100:.{d}f}%" if b else "—"


def disp(f):
    return DISPLAY.get(f, f"-{f}")


def prev_class(tok):
    if tok is None:
        return "(なし)"
    if bt(tok.tag) == "VCP":
        return "이다"
    ch = tok.form[-1]
    if not ("가" <= ch <= "힣"):
        return f"その他({bt(tok.tag)})"
    return ("子音語幹" if (ord(ch) - 0xAC00) % 28 else "母音語幹") + f"({bt(tok.tag)})"


def key_eu(f):
    """으・은・을・읍・는(と 느・습)の有無を無視したキー"""
    for p, r in (("으", ""), ("느", ""), ("을", "ㄹ"), ("은", "ㄴ"), ("는", "ㄴ"), ("읍", "ㅂ"), ("습", "ㅂ")):
        if f.startswith(p) and len(f) > 1:
            return r + f[len(p):]
    return f


def key_yo(f):
    if f == "죠":
        return "지"
    return f[:-1] if f.endswith("요") and len(f) > 1 else f


class Res:
    def __init__(self):
        self.lines = 0
        self.morph = 0
        self.morph_all = 0
        self.cat = {"v1": Counter(), "final": Counter()}
        self.ef = {"v1": Counter(), "final": Counter()}
        self.yo = {"v1": Counter(), "final": Counter()}
        self.eps = {"v1": defaultdict(set), "final": defaultdict(set)}
        self.ec = Counter()
        self.raw = {"EF": Counter(), "EC": Counter(), "EP": Counter()}
        self.raw_prev = {"EF": defaultdict(Counter), "EC": defaultdict(Counter), "EP": defaultdict(Counter)}
        self.eyo_other = []
        self.unclassified = []
        self.ep = Counter()
        self.ep_lines = defaultdict(int)
        self.ge = Counter()

    def merge(self, o):
        self.lines += o.lines
        self.morph += o.morph
        self.morph_all += o.morph_all
        for v in ("v1", "final"):
            self.cat[v] += o.cat[v]
            self.ef[v] += o.ef[v]
            self.yo[v] += o.yo[v]
            for k, s in o.eps[v].items():
                self.eps[v][k] |= s
        self.ec += o.ec


def ef_final(core, end, last, yo_tail, R, keep):
    """final の行末:(区分, EF形, 요付き)"""
    tag = bt(last.tag)
    prev = core[end - 1] if end >= 1 else None
    if tag == "EF":
        if nf(last.form) == "요":
            pt = bt(prev.tag) if prev else "(なし)"
            if prev is not None and pt in ("EF", "EC"):
                f, _ = strip_yo(prev.form)
                f = (F_EF if pt == "EF" else F_EC).get(f, f)
                return pt, f, True
            if prev is not None and pt in NOUNISH:
                return "語尾なし", None, False
            if keep:
                R.unclassified.append(pt)
            return "未区分", None, False
        f, yo = strip_yo(last.form)
        if f in ("에요", "예요"):
            pt = bt(prev.tag) if prev else "(なし)"
            if pt in ("VCP", "VCN"):
                return "EF", "야", True
            R.eyo_other.append(pt)
            return "EF", f, True
        return "EF", F_EF.get(f, f), yo or yo_tail
    if tag == "EC":
        f, yo = strip_yo(last.form)
        return "EC", F_EC.get(f, f), yo or yo_tail
    if tag.startswith("E"):
        return "その他", None, False
    return "語尾なし", None, False


def analyze(kiwi, items, store=False):
    R = Res()
    data = [(ep, row, clean(t)) for ep, row, t in items]
    data = [x for x in data if HANGUL.search(x[2])]
    R.lines = len(data)
    store_rows = []
    for (ep, row, t), toks in zip(data, kiwi.tokenize([x[2] for x in data])):
        R.morph_all += len(toks)
        core = [x for x in toks if not x.tag.startswith("S")]
        R.morph += len(core)
        seen_ep = set()
        for k, x in enumerate(core):
            tg = bt(x.tag)
            if tg == "EP":
                f = nf(x.form)
                R.raw["EP"][f] += 1
                R.raw_prev["EP"][f][prev_class(core[k - 1] if k else None)] += 1
                g = F_EP.get(f, f)
                R.ep[g] += 1
                if g not in seen_ep:
                    R.ep_lines[g] += 1
                    seen_ep.add(g)
            if tg == "EF" and nf(x.form) == "거든":
                R.ge["거든_any"] += 1
            if tg == "EF" and nf(x.form) == "거든요":
                R.ge["거든요_any"] += 1
        rec = {"ep": ep, "row": row, "t": t, "an": an(core), "cat": "語尾なし", "f": None, "core": core}
        if not core:
            if store:
                store_rows.append(rec)
            continue
        end = len(core) - 1
        yo_tail = False
        if end > 0 and core[end].form == "요" and core[end].tag in ("JX", "MM"):
            yo_tail = True
            end -= 1
        last = core[end]
        tag = bt(last.tag)
        prev = core[end - 1] if end >= 1 else None
        # 生の形(PART1)
        if tag in ("EF", "EC"):
            f = nf(last.form)
            R.raw[tag][f] += 1
            R.raw_prev[tag][f][prev_class(prev)] += 1
        # v1
        if tag.startswith("E"):
            k1 = tag if tag in ("EF", "EC") else "その他"
            if k1 == "EF":
                f, yo = strip_yo(last.form)
                f = V1_EF.get(f, f)
                R.ef["v1"][f] += 1
                R.yo["v1"][f] += int(yo or yo_tail)
                R.eps["v1"][f].add(ep)
        else:
            k1 = "語尾なし"
        R.cat["v1"][k1] += 1
        # final
        k2, f2, yo2 = ef_final(core, end, last, yo_tail, R, True)
        R.cat["final"][k2] += 1
        if k2 == "EF":
            R.ef["final"][f2] += 1
            R.yo["final"][f2] += int(yo2)
            R.eps["final"][f2].add(ep)
            if f2 == "거든":
                R.ge["거든_final"] += 1
        elif k2 == "EC":
            R.ec[f2] += 1
        rec.update(cat=k2, f=f2)
        if store:
            store_rows.append(rec)
    return R, store_rows


def avg_ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** .5
    sy = sum((y - my) ** 2 for y in ys) ** .5
    return sxy / (sx * sy) if sx and sy else None


def wcsv(path, head, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gose", required=True)
    ap.add_argument("--bts", required=True)
    ap.add_argument("--kpop", default=os.path.expanduser("~/Downloads/書籍/kpop_analyze.py"))
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="out_endings_final")
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"{a.out} に既にファイルがあります。上書きしないため停止します。")
    os.makedirs(a.out, exist_ok=True)
    O = [f"# まとめ方 final\n\nkiwipiepy {kiwipiepy.__version__} / 前処理 V2\n"]
    kiwi = Kiwi()
    rnd = random.Random(SEED)

    with open(a.gose, encoding="utf-8-sig", newline="") as f:
        g_items = [(r.get("episode", ""), i, r.get("text") or "") for i, r in enumerate(csv.DictReader(f), 1)]
    with open(a.bts, encoding="utf-8-sig", newline="") as f:
        b_raw = list(csv.DictReader(f))
    res = {}
    res["GOSE"], grows = analyze(kiwi, g_items, store=True)
    for s in ("run", "bomb"):
        res[s], _ = analyze(kiwi, [(r.get("clip_id", ""), r.get("line_no", ""), r.get("text") or "")
                                   for r in b_raw if r.get("source") == s])
    comb = Res()
    comb.merge(res["run"])
    comb.merge(res["bomb"])
    res["run+bomb"] = comb
    G = res["GOSE"]

    # ================= PART0
    O.append("## PART0 再現確認(v1)\n\n| コーパス | 項目 | 期待値 | 今回 | 一致 |\n|---|---|---:|---:|---|")
    ok = True
    for k, exp in P0.items():
        X = res[k]
        got = {"全行": X.lines, "EF": X.cat["v1"]["EF"], "EC": X.cat["v1"]["EC"], "その他": X.cat["v1"]["その他"],
               "語尾なし": X.cat["v1"]["語尾なし"], "14種類": sum(X.ef["v1"][f] for f in GOSE14_V1)}
        for lab, e in exp.items():
            O.append(f"| {k} | {lab} | {e} | {got[lab]} | {'○' if got[lab] == e else '×'} |")
            ok &= got[lab] == e
    if not ok:
        O.append("\n**不一致のため PART1 以降は実行していない。**")
        return finish(a, O)
    O.append("\n一致。")

    # ================= PART1
    O.append("\n## PART1 形違いの一覧(まとめる前)\n")
    covered = set(F_EF) | set(F_EC) | set(F_EP) | {"ㅂ니다", "을까"}
    rows = []
    uncovered = []
    for k in ("GOSE", "run", "bomb"):
        X = res[k]
        for cat in ("EF", "EC", "EP"):
            for kind, kf in (("으・은・을・읍・는", key_eu), ("요", key_yo)):
                groups = defaultdict(list)
                for f in X.raw[cat]:
                    groups[kf(f)].append(f)
                for gk, fs in groups.items():
                    if len(fs) < 2:
                        continue
                    for f in sorted(fs, key=lambda x: -X.raw[cat][x]):
                        pc = ", ".join(f"{c} {n}" for c, n in X.raw_prev[cat][f].most_common(4))
                        rows.append([k, cat, kind, gk, f, X.raw[cat][f], pc])
                        if kind != "요" and f != gk and f not in covered and (cat, f) not in [(u[1], u[2]) for u in uncovered]:
                            uncovered.append((k, cat, f, gk, X.raw[cat][f]))
    wcsv(os.path.join(a.out, "part1_variant_pairs.csv"), ["コーパス", "区分", "違い", "キー", "形", "件数", "直前の内訳(上位4)"], rows)
    O.append(f"- 組の一覧: part1_variant_pairs.csv({len(rows)}行)")
    O.append("\n規則の表にない 으・은・을・읍・는 の組(まとめていない。件数は最初に見つかったコーパスの値)\n\n| コーパス | 区分 | 形 | キー | 件数 |\n|---|---|---|---|---:|")
    for u in sorted(uncovered, key=lambda u: -u[4])[:60]:
        O.append(f"| {u[0]} | {u[1]} | {u[2]} | {u[3]} | {u[4]} |")

    # ================= PART3
    O.append("\n## PART3 final で再集計\n")
    g14f = [f for f, _ in G.ef["final"].most_common(14)]
    O.append("GOSE の上位14(final): " + "、".join(f"{f}({disp(f)})" for f in g14f))
    O.append(f"\n- v1 にあって final にない: {[f for f in GOSE14_V1 if f not in g14f] or 'なし'}")
    O.append(f"- final にあって v1 にない: {[f for f in g14f if f not in GOSE14_V1] or 'なし'}\n")
    summ = {}
    for k in ("GOSE", "run", "bomb", "run+bomb"):
        X = res[k]
        n = X.lines
        O.append(f"### {k}(全行 {n})\n\n| 区分 | v1 行数 | v1 割合 | final 行数 | final 割合 |\n|---|---:|---:|---:|---:|")
        for c in CATS:
            O.append(f"| {c} | {X.cat['v1'][c]} | {pct(X.cat['v1'][c], n)} | {X.cat['final'][c]} | {pct(X.cat['final'][c], n)} |")
        ef = X.cat["final"]["EF"]
        O.append(f"\nEF上位30(final、分母:EF行 {ef})\n\n| 順位 | 形 | 表記 | 件数 | EF比 | 累積 | 요付き | 話数 |\n|---:|---|---|---:|---:|---:|---:|---:|")
        cum = 0
        rr = []
        for i, (f, c) in enumerate(X.ef["final"].most_common(30), 1):
            cum += c
            O.append(f"| {i} | {f} | {disp(f)} | {c} | {pct(c, ef)} | {pct(cum, ef)} | {X.yo['final'][f]} | {len(X.eps['final'][f])} |")
            rr.append([i, f, disp(f), c, c / ef, cum / ef, X.yo["final"][f], len(X.eps["final"][f])])
        wcsv(os.path.join(a.out, f"ef_top30_final_{k}.csv"), ["順位", "形", "表記", "件数", "EF比", "累積", "요付き", "話数"], rr)
        for ver, g14 in (("v1", GOSE14_V1), ("final", g14f)):
            efv = X.cat[ver]["EF"]
            summ[(k, ver)] = (sum(X.ef[ver][f] for f in g14), sum(c for _, c in X.ef[ver].most_common(14)), efv)
        O.append("\n| 占有率 | v1 | final |\n|---|---:|---:|")
        for lab, idx in (("GOSEの14種類", 0), ("自身の上位14種類", 1)):
            v1, v2 = summ[(k, "v1")], summ[(k, "final")]
            O.append(f"| {lab} | {v1[idx]}/{v1[2]} = {pct(v1[idx], v1[2])} | {v2[idx]}/{v2[2]} = {pct(v2[idx], v2[2])} |")
        O.append(f"\n行末EC上位10(final、分母:EC行 {X.cat['final']['EC']}): "
                 + "、".join(f"{f}({disp(f)}) {c}" for f, c in X.ec.most_common(10)) + "\n")
    O.append("### 重なりと判定\n\n| 比較 | 版 | 上位14入り | 上位20入り | スピアマン(和集合内で順位付け直し) | 占有率 | 基準(GOSE値−5) | 判定 |\n|---|---|---:|---:|---:|---:|---:|---|")
    for k in ("run", "bomb", "run+bomb"):
        X = res[k]
        for ver, g14 in (("v1", GOSE14_V1), ("final", g14f)):
            b14 = [f for f, _ in X.ef[ver].most_common(14)]
            b20 = [f for f, _ in X.ef[ver].most_common(20)]
            i14, i20 = sum(f in b14 for f in g14), sum(f in b20 for f in g14)
            union = list(dict.fromkeys(g14 + b14))
            rho = spearman(avg_ranks([G.ef[ver][f] for f in union]), avg_ranks([X.ef[ver][f] for f in union]))
            s, _, efv = summ[(k, ver)]
            gs, _, gef = summ[("GOSE", ver)]
            thr = gs / gef * 100 - 5
            occ = s / efv * 100
            c1, c2 = occ >= thr, i20 >= 12
            v = ("成立" if c1 and c2 else "部分的に成立" if c1 or c2 else "不成立") + ("" if k == "run" else "(参考)")
            O.append(f"| GOSE×{k} | {ver} | {i14}/14 | {i20}/14 | {'—' if rho is None else f'{rho:.3f}'} | {occ:.1f}% | {thr:.1f}% | {v} |")
    O.append("\n### 先語末語尾(GOSE、final:으시→시)\n\n| EP | 件数 | 行数 | 全行に対する行の割合 |\n|---|---:|---:|---:|")
    for f in ("었", "었었", "겠", "시"):
        O.append(f"| {f} | {G.ep[f]} | {G.ep_lines[f]} | {pct(G.ep_lines[f], G.lines, 2)} |")
    gv1, gvf = summ[("GOSE", "v1")], summ[("GOSE", "final")]
    ec5_1 = 6152
    ec5_f = sum(c for _, c in G.ec.most_common(5))
    O.append("\n### 企画書v3 の数値 v1 → final\n\n| 項目 | 分母 | v1 | final |\n|---|---|---:|---:|")
    O.append(f"| 上位14の占有率(87%) | 行末がEFの行 | {gv1[0]}/{gv1[2]} = {pct(gv1[0], gv1[2])} | {gvf[0]}/{gvf[2]} = {pct(gvf[0], gvf[2])} |")
    O.append(f"| 上位14の全行比(43%) | 全行 {G.lines} | {pct(gv1[0], G.lines)} | {pct(gvf[0], G.lines)} |")
    O.append(f"| 어 の割合(35%) | 行末がEFの行 | {G.ef['v1']['어']}/{gv1[2]} = {pct(G.ef['v1']['어'], gv1[2])} | {G.ef['final']['어']}/{gvf[2]} = {pct(G.ef['final']['어'], gvf[2])} |")
    s1 = gv1[0] + ec5_1 + G.cat["v1"]["語尾なし"]
    sf = gvf[0] + ec5_f + G.cat["final"]["語尾なし"]
    O.append(f"| 上位14+行末EC上位5+語尾なし(87%) | 全行 {G.lines} | {s1} = {pct(s1, G.lines)}(14は形違いをまとめた値、EC5は既存の6,152) | {sf} = {pct(sf, G.lines)} |")
    O.append(f"| 行末が어/아の行(活用・語幹の表の分母) | — | {G.ef['v1']['어']} | {G.ef['final']['어']} |")
    O.append("\n累積カバー率の列(行末がEFの行、上位14)\n\n| 順位 | v1 | final |\n|---:|---|---|")
    c1 = c2 = 0
    l1, l2 = G.ef["v1"].most_common(14), G.ef["final"].most_common(14)
    for i in range(min(14, len(l1), len(l2))):
        c1 += l1[i][1]
        c2 += l2[i][1]
        O.append(f"| {i + 1} | {l1[i][0]} {pct(c1, gv1[2])} | {l2[i][0]} {pct(c2, gvf[2])} |")
    O.append(f"\n要確認: 直前が VCP/VCN 以外の 에요/예요(別項目のまま) GOSE {len(G.eyo_other)} 件 "
             f"{dict(Counter(G.eyo_other))} / 単独の 요 で未区分 GOSE {len(G.unclassified)} 件 {dict(Counter(G.unclassified))}")
    unc = [r for r in grows if r["cat"] == "未区分"]
    wcsv(os.path.join(a.out, "unclassified_yo_GOSE.csv"), ["話", "行番号", "行", "解析"],
         [[r["ep"], r["row"], r["t"], r["an"]] for r in rnd.sample(unc, min(30, len(unc)))])

    # ================= PART4
    O.append("\n## PART4 그래 などの区分(GOSE)\n\n| 語 | 行数 | 語尾ありとして数えた(EF/EC) | 語尾なしとして数えた | その他・未区分 |\n|---|---:|---:|---:|---:|")
    p4 = []
    for w in PART4_WORDS:
        hit = [r for r in grows if re.sub(r"[^가-힣 ]", "", r["t"]).strip().split(" ")[-1:] == [w]]
        e = [r for r in hit if r["cat"] in ("EF", "EC")]
        nn = [r for r in hit if r["cat"] == "語尾なし"]
        O.append(f"| {w} | {len(hit)} | {len(e)} | {len(nn)} | {len(hit) - len(e) - len(nn)} |")
        for lab, lst in (("語尾あり", e), ("語尾なし", nn)):
            for r in rnd.sample(lst, min(10, len(lst))):
                p4.append([w, lab, r["ep"], r["row"], r["t"], r["an"]])
    wcsv(os.path.join(a.out, "part4_geurae_examples.csv"), ["語", "区分", "話", "行番号", "行", "解析"], p4)
    O.append("\n用例: part4_geurae_examples.csv(各区分10行まで)")

    # ================= PART5
    eo = [i for i, r in enumerate(grows) if r["cat"] == "EF" and r["f"] == "어"]
    s5 = []
    for i in sorted(rnd.sample(eo, min(30, len(eo)))):
        r = grows[i]
        nx = grows[i + 1]["t"] if i + 1 < len(grows) and grows[i + 1]["ep"] == r["ep"] else ""
        s5.append([r["ep"], r["row"], r["t"], nx, ""])
    wcsv(os.path.join(a.out, "sample_check_eo_linebreak.csv"), ["話", "行番号", "行", "次の行", "判定(文末/次の行に続く/不明。空欄=未検収)"], s5)
    s6 = []
    for w in DL_FORMS:
        lst = [r for r in grows if w in r["t"]]
        for r in rnd.sample(lst, min(10, len(lst))):
            pos = r["t"].find(w)
            lem = ""
            for x in r["core"]:
                if x.start <= pos < x.start + max(x.len, 1) and bt(x.tag) in ("VV", "VA", "VX"):
                    lem = x.form + "다"
                    break
            s6.append([w, r["ep"], r["row"], r["t"], lem, "", ""])
    wcsv(os.path.join(a.out, "sample_check_d_vs_l.csv"), ["形", "話", "行番号", "行", "kiwiの辞書形", "判定(正しい/誤り。空欄=未検収)", "正しい辞書形"], s6)
    O.append(f"\n## PART5 検収用サンプル\n\n- sample_check_eo_linebreak.csv: {len(s5)}行(母数 {len(eo)}行)\n- sample_check_d_vs_l.csv: {len(s6)}行")

    # ================= PART6
    O.append("\n## PART6 記録の食い違い\n\n### 1. Run BTS の年の情報\n")
    cands = []
    for dp, dn, fn in os.walk(a.root):
        if ".venv" in dp or "site-packages" in dp:
            continue
        for f in fn:
            if f.endswith((".py", ".csv", ".json", ".md")) and ("bts" in dp.lower() or "bts" in f.lower()):
                p = os.path.join(dp, f)
                try:
                    if os.path.getsize(p) > 50_000_000:
                        continue
                    s = open(p, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                if "2015" in s and ("2021" in s or "year" in s or "通時" in s or "연도" in s):
                    cands.append(p)
    run_clips = set(r.get("clip_id") for r in b_raw if r.get("source") == "run")
    maps = []
    for p in cands:
        try:
            if p.endswith(".csv"):
                rs = list(csv.DictReader(open(p, encoding="utf-8-sig", errors="ignore")))
                if not rs:
                    continue
                cols = [c for c in rs[0].keys() if isinstance(c, str) and c]
                ycols = [c for c in cols if re.search(r"year|年|연도", c, re.I)]
                for ic in cols:
                    if len(run_clips & set(str(r.get(ic)) for r in rs)) >= .9 * len(run_clips):
                        for yc in ycols:
                            m = {str(r.get(ic)): int(y.group(1)) for r in rs
                                 for y in [re.search(r"(20\d\d)", str(r.get(yc, "")))] if y}
                            if run_clips <= set(m):
                                maps.append((p, ic, yc))
        except Exception:
            continue
    O.append(f"候補ファイル {len(cands)} 件。Run BTS の全クリップに年(列名に year/年/연도)を割り当てられる CSV: {len(maps)} 件")
    for m in maps:
        O.append(f"- {m[0]}(クリップ列 {m[1]}、年の列 {m[2]})")
    O.append("\n年別の占有率: 未実施(" + ("候補が1件だけ見つかったが、正本かの確認が必要なため自動では使わない" if len(maps) == 1
                                  else "候補が複数で正本を決められない" if maps else "見つからない") + ")")

    O.append("\n### 2. 形態素数の分母\n\n| コーパス | 前処理 | 記号除く | 記号含む | 行数 |\n|---|---|---:|---:|---:|")
    v0 = [re.sub(r"\s+", " ", t).strip() for _, _, t in g_items]
    v0 = [t for t in v0 if HANGUL.search(t)]
    n_ex = n_all = 0
    for toks in kiwi.tokenize(v0):
        n_all += len(toks)
        n_ex += sum(1 for x in toks if not x.tag.startswith("S"))
    O.append(f"| GOSE corpus.csv | V2 | {G.morph} | {G.morph_all} | {G.lines} |")
    O.append(f"| GOSE corpus.csv | V0(空白のみ、テロップ含む) | {n_ex} | {n_all} | {len(v0)} |")
    O.append(f"| Run BTS | V2 | {res['run'].morph} | {res['run'].morph_all} | {res['run'].lines} |")
    O.append(f"| BOMB | V2 | {res['bomb'].morph} | {res['bomb'].morph_all} | {res['bomb'].lines} |")
    O.append("| 以前(GOSE) | utterances.json・kpop_analyze.clean_text・記号含む | — | 約821,000 | — |")
    O.append("| 以前(BTS) | Run BTS の .srt 144本・記号含む(scene_extraction.py) | — | 約727,000 | — |")
    O.append("\n- kpop_analyze.clean_text による旧方式の再現: 未実施(kpop_analyze.py を実行しない指示のため)")

    O.append("\n### 3. 거든 の倍率(GOSE ÷ Run BTS、1万形態素あたり)\n")
    Rr = res["run"]
    dens = (("V2・記号除く", G.morph, Rr.morph), ("V2・記号含む", G.morph_all, Rr.morph_all),
            ("GOSE 以前の821,000 / Run V2・記号含む", 821000, Rr.morph_all))
    defs = (("요なしの거든・位置問わず(EF)", lambda X: X.ge["거든_any"]),
            ("거든요のみ・位置問わず(EF)", lambda X: X.ge["거든요_any"]),
            ("両方・行末のみ(final)", lambda X: X.ge["거든_final"]),
            ("両方・位置問わず(EF)", lambda X: X.ge["거든_any"] + X.ge["거든요_any"]))
    O.append("| 数え方 | 分母 | GOSE 件数 | GOSE/1万 | Run 件数 | Run/1万 | 比 |\n|---|---|---:|---:|---:|---:|---:|")
    for lab, fn in defs:
        gc, rc = fn(G), fn(Rr)
        for dl, gd, rd in dens:
            gp, rp = gc / gd * 1e4, rc / rd * 1e4
            O.append(f"| {lab} | {dl}({gd} / {rd}) | {gc} | {gp:.2f} | {rc} | {rp:.2f} | {gp / rp:.2f} |" if rp else
                     f"| {lab} | {dl} | {gc} | {gp:.2f} | {rc} | 0 | — |")
    O.append("\n以前の値: 요なしの거든 3.92倍、거든요 GOSE 3.02・BTS 3.29")

    O.append("\n### 4. kpop_analyze.py(読むだけ)\n")
    kp = os.path.expanduser(a.kpop)
    if os.path.exists(kp):
        st = os.stat(kp)
        O.append(f"- {kp}: {st.st_size} バイト、{time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")
        src = open(kp, encoding="utf-8", errors="ignore").read().splitlines()
        idx = [i for i, l in enumerate(src) if re.search(r"VV-I|VA-I|VV-R|VA-R|split\(['\"]-['\"]\)|\.tag\s*(==|!=|in|not in)", l)]
        O.append("\n文字列検索(VV-I/VA-I/VV-R/VA-R/split('-')/tag の比較)に当たった行(前後2行):\n\n```")
        shown = set()
        for i in idx:
            for j in range(max(0, i - 2), min(len(src), i + 3)):
                if j not in shown:
                    O.append(f"{j + 1}: {src[j]}")
                    shown.add(j)
            O.append("...")
        O.append("```")
    else:
        O.append(f"- {kp}: ファイルなし")
    return finish(a, O)


def finish(a, O):
    txt = "\n".join(O)
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
