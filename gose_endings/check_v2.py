#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""企画書v2 修正用の検算(既存の集計出力 + 야 の再解析)

使い方(コーパス分析フォルダで):
  python3 check_v2.py --corpus seventeen_analysis/output/ani_textbook_pipeline/corpus.csv --out out_endings
出力: 画面に検算結果、out_endings/ya_breakdown_sample.csv(判定欄は空欄)
"""
import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict

BR = re.compile(r"\[[^\]]*\]")
PA = re.compile(r"\([^)]*\)")
HANGUL = re.compile(r"[가-힣]")


def clean(t):
    return re.sub(r"\s+", " ", PA.sub(" ", BR.sub(" ", t))).strip()


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="out_endings")
    a = ap.parse_args()
    o = a.out

    # 修正2: 行末の内訳と EC 上位10(endings_line_final.csv / line_final_no_ending.csv)
    fin = rows(os.path.join(o, "endings_line_final.csv"))
    by = Counter()
    for r in fin:
        by[r["区分"]] += int(r["件数"])
    ne = sum(int(r["件数"]) for r in rows(os.path.join(o, "line_final_no_ending.csv")) if r["形"] == "(全体)")
    print("## 修正2 行末の内訳 (endings_line_final.csv の区分別合計 + line_final_no_ending.csv の(全体)行)")
    for k in ("EF", "EC", "その他"):
        print(f"- {k}: {by[k]}")
    print(f"- 語尾なし: {ne}")
    print(f"- 合計: {sum(by.values()) + ne}")
    print("\n## 修正2 行末EC 上位10 (endings_line_final.csv, タグ=EC)")
    ec = [r for r in fin if r["タグ"] == "EC"]
    print(f"- EC合計 {sum(int(r['件数']) for r in ec)}")
    for i, r in enumerate(ec[:10], 1):
        print(f"{i} {r['形']} 件数{r['件数']} 요付{r['요付き件数']} 話数{r['出現話数']}")

    # 修正2: 行分割の検査の実施状況 / 修正4: 音声照合の記録
    for name, col in (("sample_check_line_final_EC.csv", "判定(文末/次行に続く/不明)"),
                      ("sample_audio_check_30.csv", "照合メモ"),
                      ("sample_audio_check_30_url.csv", "照合メモ")):
        p = os.path.join(o, name)
        if os.path.exists(p):
            rs = rows(p)
            filled = [r for r in rs if (r.get(col) or "").strip()]
            print(f"\n## {name}: {len(rs)}行中 記入済み {len(filled)}行")
            c = Counter((r.get(col) or "").strip() for r in filled)
            for k, v in c.most_common(10):
                print(f"  - {k[:40]}: {v}")
        else:
            print(f"\n## {name}: ファイルなし")

    # 修正6: 助詞表の 아/야
    print("\n## 修正6 particles_J.csv の 아・야 の行")
    for r in rows(os.path.join(o, "particles_J.csv")):
        if r["形"] in ("아", "야"):
            print(f"- {r['形']} 件数{r['件数']} タグ{r['タグ']}")

    # 修正6: 行末 야/EF の内訳(再解析)
    from kiwipiepy import Kiwi
    import kiwipiepy
    lines = []
    with open(a.corpus, encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            t = clean(r.get("text") or "")
            if HANGUL.search(t):
                lines.append((r.get("episode", ""), i, t))
    kiwi = Kiwi()
    cat = defaultdict(list)
    for (ep, row, t), toks in zip(lines, kiwi.tokenize([x[2] for x in lines])):
        core = [x for x in toks if not x.tag.startswith("S")]
        if not core:
            continue
        end = len(core) - 1
        if end > 0 and core[end].form == "요" and core[end].tag in ("JX", "MM"):
            end -= 1
        e = core[end]
        if not (e.tag == "EF" and e.form == "야"):
            continue
        p1 = core[end - 1] if end >= 1 else None
        p2 = core[end - 2] if end >= 2 else None
        if p1 is not None and p1.tag == "VCP":
            if p2 is not None and p2.tag == "NNP":
                k = "呼びかけ疑い(固有名詞+이+야)"
            elif end == 2 and p2 is not None and p2.tag == "NNG":
                k = "呼びかけ疑い(一般名詞だけの行)"
            else:
                k = "이야(指定詞)"
        else:
            k = "その他(指定詞以外)"
        cat[k].append((ep, row, t, " ".join(f"{x.form}/{x.tag}" for x in core)))
    print(f"\n## 修正6 行末 야/EF の内訳 (kiwipiepy {kiwipiepy.__version__}, 前処理V2)")
    tot = sum(len(v) for v in cat.values())
    print(f"- 合計 {tot}")
    rnd = random.Random(42)
    out = []
    for k in sorted(cat, key=lambda x: -len(cat[x])):
        v = cat[k]
        print(f"- {k}: {len(v)}")
        for ep, row, t, an in sorted(rnd.sample(v, min(10, len(v))), key=lambda x: (x[0], x[1])):
            out.append([k, ep, row, t, an, ""])
    with open(os.path.join(o, "ya_breakdown_sample.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["区分", "話", "行番号", "行", "解析", "判定(空欄=未検収)"])
        w.writerows(out)
    print(f"- KWIC各10行 → {o}/ya_breakdown_sample.csv")


if __name__ == "__main__":
    main()
