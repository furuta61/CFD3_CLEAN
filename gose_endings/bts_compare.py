#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三弾 企画判断用：行末語尾の上位が GOSE と BTS で重なるか

GOSE の集計(part1_4_endings.py)と同じ処理を両方にかける。
  前処理V2 / 行末=記号を除いた最後のトークン(末尾の 요 JX/MM は1つ手前) /
  4区分(EF・EC・その他の語尾・語尾なし) / 요の有無は同じ項目 /
  形違いのまとめは ㅂ니다→습니다、을까→ㄹ까 のみ
既存の出力は上書きしない(出力先は --out、既定 out_endings_bts)。

使い方(コーパス分析フォルダで):
  python3 bts_compare.py --gose seventeen_analysis/output/ani_textbook_pipeline/corpus.csv \
      --bts bts_pipeline/bts_textbook_pipeline/bts_corpus.csv --titles bts_pipeline/kikaku
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict

import kiwipiepy
from kiwipiepy import Kiwi

BR = re.compile(r"\[[^\]]*\]")
PA = re.compile(r"\([^)]*\)")
HANGUL = re.compile(r"[가-힣]")
JAMO = str.maketrans({"ᆫ": "ㄴ", "ᆯ": "ㄹ", "ᆸ": "ㅂ", "ᆷ": "ㅁ", "ᆻ": "ㅆ"})
NO_STRIP = {"세요", "에요", "예요"}
MERGE = {"ㅂ니다": "습니다", "을까": "ㄹ까"}
GOSE14 = ["어", "야", "지", "습니다", "다", "네", "잖아", "세요", "자", "ㄹ게", "ㄹ까", "ㄴ다", "ㄴ가", "거든"]
PART0 = {"全行": 102006, "EF": 50783, "EC": 11090, "その他": 891, "語尾なし": 39242, "上位14": 44020}
SEED = 42


# ---- part1_4_endings.py と同じ関数 ----
def clean(t):
    return re.sub(r"\s+", " ", PA.sub(" ", BR.sub(" ", t))).strip()


def nf(form):
    return form.translate(JAMO)


def strip_yo(form):
    form = nf(form)
    if form == "죠":
        return "지", True
    if form in NO_STRIP:
        return form, True
    if form.endswith("요") and len(form) > 1 and form not in NO_STRIP:
        return form[:-1], True
    return form, False


def analyze(kiwi, items):
    """items: [(episode, text_raw)] → 集計 dict"""
    lines = [(ep, clean(t)) for ep, t in items]
    lines = [(ep, t) for ep, t in lines if HANGUL.search(t)]
    cat = Counter()
    ef, ef_yo, ef_eps = Counter(), Counter(), defaultdict(set)
    morph = 0
    for (ep, t), toks in zip(lines, kiwi.tokenize([x[1] for x in lines])):
        core = [x for x in toks if not x.tag.startswith("S")]
        morph += len(core)
        if not core:
            continue
        end = len(core) - 1
        yo_tail = False
        if end > 0 and core[end].form == "요" and core[end].tag in ("JX", "MM"):
            yo_tail = True
            end -= 1
        last = core[end]
        tag = last.tag.split("-")[0]
        if tag.startswith("E"):
            kind = tag if tag in ("EF", "EC") else "その他"
            cat[kind] += 1
            if kind == "EF":
                f, yo = strip_yo(last.form)
                f = MERGE.get(f, f)
                ef[f] += 1
                ef_yo[f] += int(yo or yo_tail)
                ef_eps[f].add(ep)
        else:
            cat["語尾なし"] += 1
    return {"lines": len(lines), "cat": cat, "ef": ef, "ef_yo": ef_yo, "ef_eps": ef_eps, "morph": morph}


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "—"


def share(res, forms):
    tot = res["cat"]["EF"]
    return sum(res["ef"][f] for f in forms), tot


def rank_map(res):
    return {f: i for i, (f, _) in enumerate(res["ef"].most_common(), 1)}


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** .5
    sy = sum((y - my) ** 2 for y in ys) ** .5
    return sxy / (sx * sy) if sx and sy else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gose", required=True)
    ap.add_argument("--bts", required=True)
    ap.add_argument("--titles", help="runbts_titles.json / bomb_titles.json のあるフォルダ")
    ap.add_argument("--out", default="out_endings_bts")
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"{a.out} に既にファイルがあります。上書きしないため停止します。")
    os.makedirs(a.out, exist_ok=True)
    R = [f"# GOSE×BTS 行末語尾の重なり\n\nkiwipiepy {kiwipiepy.__version__}\n"]
    kiwi = Kiwi()

    # ---------------- PART0
    with open(a.gose, encoding="utf-8-sig", newline="") as f:
        g_items = [(r.get("episode", ""), r.get("text") or "") for r in csv.DictReader(f)]
    G = analyze(kiwi, g_items)
    got = {"全行": G["lines"], "EF": G["cat"]["EF"], "EC": G["cat"]["EC"], "その他": G["cat"]["その他"],
           "語尾なし": G["cat"]["語尾なし"], "上位14": share(G, GOSE14)[0]}
    R.append("## PART0 再現確認(GOSE corpus.csv)\n\n| 項目 | 期待値 | 今回 | 一致 |\n|---|---:|---:|---|")
    ok = True
    for k, v in PART0.items():
        R.append(f"| {k} | {v} | {got[k]} | {'○' if got[k] == v else '×'} |")
        ok &= got[k] == v
    if not ok:
        R.append("\n**不一致のため PART1 以降は実行していない。**")
        return finish(a, R)
    R.append("\n一致。")

    # ---------------- PART1
    R.append("\n## PART1 BTSコーパスの所在\n")
    try:
        cands = subprocess.run(["mdfind", "-name", "bts_corpus.csv"], capture_output=True, text=True).stdout.split()
        cands = [c for c in cands if c.endswith("bts_corpus.csv")]
    except FileNotFoundError:
        cands = None
    R.append(f"- 指定パス: {os.path.abspath(a.bts)}")
    R.append(f"- mdfind の候補: {cands if cands is not None else '未実施(mdfind なし)'}")
    if cands and len(cands) > 1:
        R.append("\n**候補が複数あり正本を決められないため停止。**")
        return finish(a, R)
    with open(a.bts, encoding="utf-8-sig", newline="") as f:
        b_rows = list(csv.DictReader(f))
    by_src = defaultdict(list)
    for r in b_rows:
        by_src[r.get("source", "")].append(r)
    R.append(f"- 列: {list(b_rows[0].keys()) if b_rows else []}")
    for s, rs in sorted(by_src.items()):
        R.append(f"- source={s}: {len(rs)}行 / クリップ {len(set(r.get('clip_id') for r in rs))}")
    R.append("- 元データ: build_bts_corpus.py の記述では Whisper ASR の .srt(Run BTS S1-2 56本・S3 88本、BOMB 654本)。"
             "hallu_filter・ハングル無し行除去・結合重複行除去を適用済み")
    rnd = random.Random(SEED)
    smp = rnd.sample(b_rows, min(30, len(b_rows)))
    with open(os.path.join(a.out, "sample_bts_lines_30.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source", "clip_id", "line_no", "text", "判定(空欄=未検収)"])
        for r in smp:
            w.writerow([r.get("source"), r.get("clip_id"), r.get("line_no"), r.get("text"), ""])
    R.append("- sample_bts_lines_30.csv を出力(判定欄は空欄)")

    # ---------------- PART2
    res = {"GOSE": G}
    for s in ("run", "bomb"):
        res[s] = analyze(kiwi, [(r.get("clip_id", ""), r.get("text") or "") for r in by_src.get(s, [])])
    res["run+bomb"] = analyze(kiwi, [(r.get("clip_id", ""), r.get("text") or "") for r in b_rows])
    R.append("\n## PART2 集計\n")
    for name in ("run", "bomb", "run+bomb"):
        X = res[name]
        n = X["lines"]
        R.append(f"### {name}(全行 {n})\n\n| 区分 | 行数 | 全行に対する割合 |\n|---|---:|---:|")
        for k in ("EF", "EC", "その他", "語尾なし"):
            R.append(f"| {k} | {X['cat'][k]} | {pct(X['cat'][k], n)} |")
        ef_tot = X["cat"]["EF"]
        R.append(f"\nEF上位30(分母:EF行 {ef_tot})\n\n| 順位 | 形 | 件数 | EF比 | 累積 | 요付き | 話数 |\n|---:|---|---:|---:|---:|---:|---:|")
        cum = 0
        rows_out = []
        for i, (f, c) in enumerate(X["ef"].most_common(30), 1):
            cum += c
            R.append(f"| {i} | {f} | {c} | {pct(c, ef_tot)} | {pct(cum, ef_tot)} | {X['ef_yo'][f]} | {len(X['ef_eps'][f])} |")
            rows_out.append([i, f, c, c / ef_tot if ef_tot else "", cum / ef_tot if ef_tot else "", X["ef_yo"][f], len(X["ef_eps"][f])])
        with open(os.path.join(a.out, f"ef_top30_{name}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["順位", "形", "件数", "EF比", "累積", "요付き", "話数"])
            w.writerows(rows_out)
        g14, _ = share(X, GOSE14)
        own14 = sum(c for _, c in X["ef"].most_common(14))
        R.append(f"\n- GOSEの14種類の占有率: {g14} / EF行 {ef_tot} = {pct(g14, ef_tot)}")
        R.append(f"- 自身の上位14種類の占有率: {own14} / EF行 {ef_tot} = {pct(own14, ef_tot)}\n")

    # ---------------- PART3
    R.append("## PART3 重なり(GOSE と各BTS)\n")
    verdict = {}
    for name in ("run", "bomb"):
        X = res[name]
        gr, br = rank_map(G), rank_map(X)
        b14 = [f for f, _ in X["ef"].most_common(14)]
        b20 = [f for f, _ in X["ef"].most_common(20)]
        in14 = sum(f in b14 for f in GOSE14)
        in20 = sum(f in b20 for f in GOSE14)
        union = list(dict.fromkeys(GOSE14 + b14))
        mx_g, mx_b = len(gr) + 1, len(br) + 1
        xs = [gr.get(f, mx_g) for f in union]
        ys = [br.get(f, mx_b) for f in union]
        rho = spearman(xs, ys)
        R.append(f"### GOSE × {name}\n")
        R.append(f"- GOSEの14のうち {name} の上位14に入る数: {in14} / 14")
        R.append(f"- GOSEの14のうち {name} の上位20に入る数: {in20} / 14")
        R.append(f"- 和集合 {len(union)} 項目のスピアマン順位相関: {'—' if rho is None else f'{rho:.3f}'}"
                 "(各コーパスの全EF順位を使用。出現しない項目は最下位+1)")
        R.append(f"\n| 形 | GOSE順位 | GOSE件数 | {name}順位 | {name}件数 |\n|---|---:|---:|---:|---:|")
        for f in union:
            R.append(f"| {f} | {gr.get(f, '—')} | {G['ef'][f]} | {br.get(f, '—')} | {X['ef'][f]} |")
        top = list(dict.fromkeys([f for f, _ in G["ef"].most_common(30)] + [f for f, _ in X["ef"].most_common(30)]))
        R.append(f"\n1万形態素あたりの比(GOSE÷{name})が2倍以上/0.5倍以下(対象:両者のEF上位30の和集合、分母:記号を除く形態素数 "
                 f"GOSE {G['morph']} / {name} {X['morph']})\n\n| 形 | GOSE/1万 | {name}/1万 | 比 |\n|---|---:|---:|---:|")
        for f in top:
            gp = G["ef"][f] / G["morph"] * 1e4 if G["morph"] else 0
            bp = X["ef"][f] / X["morph"] * 1e4 if X["morph"] else 0
            if bp == 0 or gp == 0:
                R.append(f"| {f} | {gp:.2f} | {bp:.2f} | {'∞' if bp == 0 else '0'} |")
            elif gp / bp >= 2 or gp / bp <= .5:
                R.append(f"| {f} | {gp:.2f} | {bp:.2f} | {gp / bp:.2f} |")
        s14, tot = share(X, GOSE14)
        c1 = tot > 0 and s14 / tot >= .817
        c2 = in20 >= 12
        verdict[name] = ("成立" if c1 and c2 else "部分的に成立" if c1 or c2 else "不成立", s14, tot, in20)
        R.append("")

    # ---------------- PART4
    R.append("## PART4 年別(Run BTS)\n")
    years = {}
    if a.titles:
        p = os.path.join(a.titles, "runbts_titles.json")
        if os.path.exists(p):
            for k, v in json.load(open(p, encoding="utf-8")).items():
                if isinstance(v, dict):
                    for key in ("date", "upload_date", "published", "published_at", "year"):
                        if v.get(key):
                            m = re.search(r"(20\d\d)", str(v[key]))
                            if m:
                                years[k] = int(m.group(1))
                                break
    run_rows = by_src.get("run", [])
    covered = sum(1 for r in run_rows if r.get("clip_id") in years)
    if not years or covered < len(run_rows):
        R.append(f"未実施:年の情報が得られない(年が分かった行 {covered} / Run BTS {len(run_rows)} 行)。"
                 "runbts_titles.json に日付の項目(date / upload_date / published 等)が無いか、一部のクリップに無い。")
    else:
        for lab, cond in (("2015–2016", lambda y: y <= 2016), ("2017以降", lambda y: y >= 2017)):
            X = analyze(kiwi, [(r["clip_id"], r.get("text") or "") for r in run_rows if cond(years[r["clip_id"]])])
            s14, tot = share(X, GOSE14)
            R.append(f"- {lab}: GOSEの14種類 {s14} / EF行 {tot} = {pct(s14, tot)}(全行 {X['lines']})")

    # ---------------- 判定
    R.append("\n## 判定(基準:Run BTS で GOSE14 の占有率 ≥81.7% かつ GOSE14 のうち上位20入り ≥12)\n")
    v, s14, tot, in20 = verdict["run"]
    R.append(f"- Run BTS: **{v}**(占有率 {s14}/{tot} = {pct(s14, tot)}、上位20入り {in20}/14)")
    v, s14, tot, in20 = verdict["bomb"]
    R.append(f"- BOMB(参考値・判定に使わない): 占有率 {s14}/{tot} = {pct(s14, tot)}、上位20入り {in20}/14")
    return finish(a, R)


def finish(a, R):
    txt = "\n".join(R)
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
