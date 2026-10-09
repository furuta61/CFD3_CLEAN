#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行末の 어/아(EF) の前に来る語幹と活用パターンを数える(書籍の活用章の材料)

前処理は part1_4_endings.py と同じ V2。
使い方: python3 conj_eo.py --corpus corpus.csv --out out_endings
出力: conj_types.csv / conj_stems.csv / conj_summary.md(画面にも表示)
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

from kiwipiepy import Kiwi

BR = re.compile(r"\[[^\]]*\]")
PA = re.compile(r"\([^)]*\)")
HANGUL = re.compile(r"[가-힣]")
# 아/어 の活用を使う形(同じ規則を覚えれば使える)
EO_FAMILY = {("어", "EF"), ("어요", "EF"), ("어", "EC"), ("어서", "EC"), ("어도", "EC"), ("어야", "EC"),
             ("었", "EP"), ("어라", "EF")}


def clean(t):
    return re.sub(r"\s+", " ", PA.sub(" ", BR.sub(" ", t))).strip()


def jamo(syl):
    c = ord(syl) - 0xAC00
    if not 0 <= c < 11172:
        return None, None
    return (c // 28) % 21, c % 28  # 中声番号, 終声番号


def conj_type(form, tag):
    base = tag.split("-")[0]
    if base == "VCP":
        return "指定詞 이다"
    if form == "하" and base in ("VV", "XSV", "XSA", "VX", "VA"):
        return "하다 → 해"
    last = form[-1]
    med, fin = jamo(last)
    if med is None:
        return "その他"
    if tag.endswith("-I"):
        return {17: "ㅂ不規則(덥다→더워)", 7: "ㄷ不規則(듣다→들어)", 19: "ㅅ不規則(낫다→나아)",
                27: "ㅎ不規則(그렇다→그래)"}.get(fin, "不規則その他")
    if last == "르":
        return "르不規則(모르다→몰라)"
    if fin:
        return "子音語幹(そのまま 아/어)"
    return {0: "母音同化・脱落(가다→가)", 4: "母音同化・脱落(가다→가)", 1: "母音同化・脱落(가다→가)",
            5: "母音同化・脱落(가다→가)", 6: "母音同化・脱落(가다→가)",
            8: "ㅗ縮約(보다→봐)", 13: "ㅜ縮約(주다→줘)", 20: "ㅣ縮約(마시다→마셔)",
            18: "으脱落(쓰다→써)", 11: "되다→돼"}.get(med, "母音語幹その他")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="out_endings")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    texts = []
    with open(a.corpus, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            t = clean(r.get("text") or "")
            if HANGUL.search(t):
                texts.append(t)
    kiwi = Kiwi()
    types, stems = Counter(), Counter()
    stem_type, stem_ex = {}, defaultdict(Counter)
    past = n_final = 0
    fam_lines = 0
    for t, toks in zip(texts, kiwi.tokenize(texts)):
        core = [x for x in toks if not x.tag.startswith("S")]
        if any((x.form, x.tag.split("-")[0]) in EO_FAMILY for x in core):
            fam_lines += 1
        if not core:
            continue
        end = len(core) - 1
        if end > 0 and core[end].form == "요" and core[end].tag in ("JX", "MM"):
            end -= 1
        e = core[end]
        if not (e.tag == "EF" and e.form in ("어", "어요")):
            continue
        n_final += 1
        k = end - 1
        is_past = False
        while k >= 0 and core[k].tag == "EP":
            is_past |= core[k].form in ("었", "았")
            k -= 1
        past += is_past
        if k < 0 or not (core[k].tag.startswith("V") or core[k].tag.startswith("XS")):
            types["語幹不明"] += 1
            continue
        s = core[k]
        ty = conj_type(s.form, s.tag)
        lemma = ("〜" if s.tag.startswith("XS") else "") + s.form + "다"
        types[ty] += 1
        stems[lemma] += 1
        stem_type[lemma] = ty
        stem_ex[lemma][t.split()[-1] if t.split() else t] += 1

    with open(os.path.join(a.out, "conj_types.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["活用パターン", "件数", "割合"])
        for ty, c in types.most_common():
            w.writerow([ty, c, round(c / n_final, 4)])
    cum = 0
    rows = []
    for lemma, c in stems.most_common():
        cum += c
        rows.append([lemma, stem_type[lemma], c, round(c / n_final, 4), round(cum / n_final, 4),
                     " / ".join(f"{x}({n})" for x, n in stem_ex[lemma].most_common(3))])
    with open(os.path.join(a.out, "conj_stems.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["語幹", "活用パターン", "件数", "割合", "累積", "実例(行末の語)"])
        w.writerows(rows)

    S = ["# 行末 어/아 の活用\n",
         f"- 行末が 어/아(요付き含む)の行: {n_final}(うち過去 었/았 付き {past}, {past / n_final * 100:.1f}%)",
         f"- 아/어 活用を使う形(어・어요・어서・어도・어야・었 等)を1つ以上含む行: {fam_lines} / {len(texts)}"
         f"({fam_lines / len(texts) * 100:.1f}%)\n",
         "## 活用パターン別\n", "| パターン | 件数 | 割合 |", "|---|---:|---:|"]
    for ty, c in types.most_common():
        S.append(f"| {ty} | {c} | {c / n_final * 100:.1f}% |")
    S += ["\n## 語幹 上位30(累積カバー率)\n", "| 語幹 | パターン | 件数 | 累積 | 実例 |", "|---|---|---:|---:|---|"]
    for r in rows[:30]:
        S.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[4] * 100:.1f}% | {r[5]} |")
    S.append("\n注: kiwi は 아/어 を 어 に統一して出力する。1音節の行(가・와・그래 等)は感嘆詞・名詞と解析されることがあり、ここには入らない。")
    txt = "\n".join(S)
    open(os.path.join(a.out, "conj_summary.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
