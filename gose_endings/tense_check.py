#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""時制・意志推量の語尾の再調査(GOSE corpus.csv)

前処理・行末の決め方は part1_4_endings.py と同じ(V2、記号を除いた最後のトークン、末尾の 요 JX/MM は1つ手前)。
EP の 았/었/였 は 었 にまとめる(kiwi が既に 었 に正規化している場合も、元の形を別表で記録する)。
既存の出力は上書きしない(出力先は --out、既定 out_tense)。

使い方(コーパス分析フォルダで):
  python3 tense_check.py --corpus seventeen_analysis/output/ani_textbook_pipeline/corpus.csv \
      --stems out_endings/conj_stems.csv --out out_tense
"""
import argparse
import csv
import os
import random
import re
import sys
from collections import Counter, defaultdict

import kiwipiepy
from kiwipiepy import Kiwi

BR = re.compile(r"\[[^\]]*\]")
PA = re.compile(r"\([^)]*\)")
HANGUL = re.compile(r"[가-힣]")
JAMO = str.maketrans({"ᆫ": "ㄴ", "ᆯ": "ㄹ", "ᆸ": "ㅂ", "ᆷ": "ㅁ", "ᆻ": "ㅆ"})
NO_STRIP = {"세요", "에요", "예요"}
MERGE_EF = {"ㅂ니다": "습니다", "을까": "ㄹ까"}
PRED = {"VV", "VA", "VX", "VCP", "XSV", "XSA"}
EP_NORM = {"았": "었", "였": "었", "았었": "었었", "였었": "었었"}
SEED = 42
N_EXPECTED = 102006


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


def lemma(tok):
    return ("〜" if bt(tok.tag).startswith("XS") else "") + tok.form + "다"


def an(core):
    return " ".join(f"{t.form}/{t.tag}" for t in core)


def pct(a, b):
    return f"{a / b * 100:.2f}%" if b else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--stems", help="out_endings/conj_stems.csv(既存の逆引き表の語幹順位)")
    ap.add_argument("--out", default="out_tense")
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"{a.out} に既にファイルがあります。上書きしないため停止します。")
    os.makedirs(a.out, exist_ok=True)
    R = [f"# 時制・意志推量の語尾\n\nkiwipiepy {kiwipiepy.__version__} / 前処理 V2\n"]

    lines = []
    with open(a.corpus, encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            t = clean(r.get("text") or "")
            if HANGUL.search(t):
                lines.append((r.get("episode", ""), i, t))
    N = len(lines)
    R.append(f"## PART0\n\n- 対象行数 {N}(期待値 {N_EXPECTED}): {'一致' if N == N_EXPECTED else '不一致'}")
    if N != N_EXPECTED:
        R.append("\n**不一致のため PART1 以降は実行していない。**")
        return finish(a, R)

    kiwi = Kiwi()
    ep_tok, ep_lines, ep_eps, ep_raw = Counter(), defaultdict(set), defaultdict(set), Counter()
    pred_lines = 0
    strict, broad = [], []                      # (line_idx, yo)
    eflist = Counter()                          # 行内すべての位置の EF(ㄹ게/ㄹ래/ㄹ까 系)
    final_ef = Counter()                        # 行末 EF(まとめなし、part1_4 と同じ)
    final_ef_merged = Counter()
    gyet_next = Counter()
    gyet_next_yo = Counter()
    combo = Counter()
    past_surf, past_lemma = Counter(), Counter()
    surf_lemma = {}
    all_pred = Counter()
    gyet_lines, geo_lines, eot_lines = [], [], []
    an_cache = {}

    for li, ((ep, row, t), toks) in enumerate(zip(lines, kiwi.tokenize([x[2] for x in lines]))):
        core = [x for x in toks if not x.tag.startswith("S")]
        if any(bt(x.tag) in PRED for x in core):
            pred_lines += 1
        seen = set()
        has_gyet = has_geo = has_eot = False
        for k, x in enumerate(core):
            tg = bt(x.tag)
            if tg in PRED:
                all_pred[lemma(x)] += 1
            if tg == "EP":
                ep_raw[x.form] += 1
                f = EP_NORM.get(x.form, x.form)
                ep_tok[f] += 1
                ep_lines[f].add(li)
                ep_eps[f].add(ep)
                if f == "겠":
                    has_gyet = True
                    j = k + 1
                    while j < len(core) and bt(core[j].tag) == "EP":
                        j += 1
                    if j < len(core) and bt(core[j].tag).startswith("E"):
                        nfm, yo = strip_yo(core[j].form)
                        yo = yo or (j + 1 < len(core) and core[j + 1].form == "요" and core[j + 1].tag in ("JX", "MM"))
                        key = f"{nfm}/{bt(core[j].tag)}"
                        gyet_next[key] += 1
                        gyet_next_yo[key] += int(yo)
                    else:
                        gyet_next["(語尾なし)"] += 1
                if f == "었":
                    has_eot = True
                    # 直前の用言(시 は飛ばす)
                    j = k - 1
                    while j >= 0 and bt(core[j].tag) == "EP":
                        j -= 1
                    if j >= 0 and bt(core[j].tag) in PRED:
                        s = core[j]
                        surf = t[s.start: x.start + x.len]
                        lm = lemma(s)
                        past_surf[(surf, lm)] += 1
                        past_lemma[lm] += 1
            if tg == "EF":
                fm, _ = strip_yo(x.form)
                if fm in ("ㄹ게", "을게", "ㄹ래", "을래", "ㄹ까", "을까"):
                    eflist[fm] += 1
            # -ㄹ 거
            if tg == "ETM" and nf(x.form) in ("ㄹ", "을") and k + 1 < len(core) \
                    and bt(core[k + 1].tag) == "NNB" and core[k + 1].form in ("거", "것"):
                broad.append(li)
                has_geo = True
                j = k + 2
                if j < len(core) and bt(core[j].tag) == "VCP":
                    j += 1
                while j < len(core) and bt(core[j].tag) == "EP":
                    j += 1
                if j < len(core) and bt(core[j].tag).startswith("E") and bt(core[j].tag) != "ETM":
                    _, yo = strip_yo(core[j].form)
                    yo = yo or (j + 1 < len(core) and core[j + 1].form == "요" and core[j + 1].tag in ("JX", "MM"))
                    strict.append((li, int(yo)))
        if has_gyet:
            gyet_lines.append(li)
        if has_geo:
            geo_lines.append(li)
        if has_eot:
            eot_lines.append(li)
        an_cache[li] = an(core)
        # 行末 EF(part1_4 と同じ)
        if core:
            end = len(core) - 1
            if end > 0 and core[end].form == "요" and core[end].tag in ("JX", "MM"):
                end -= 1
            e = core[end]
            if bt(e.tag) == "EF":
                fm, _ = strip_yo(e.form)
                final_ef[fm] += 1
                mf = MERGE_EF.get(fm, fm)
                final_ef_merged[mf] += 1
                j = end - 1
                chain = []
                while j >= 0 and bt(core[j].tag) == "EP":
                    chain.insert(0, EP_NORM.get(core[j].form, core[j].form))
                    j -= 1
                if chain:
                    combo[("+".join(chain), mf)] += 1

    # ---- PART1
    R.append(f"\n## PART1 先語末語尾(EP)\n\n分母①全行 {N} / ②用言(VV・VA・VX・VCP・XSV・XSA)を含む行 {pred_lines}\n")
    R.append("| EP | 件数 | 行数 | ①に対する割合 | ②に対する割合 | 話数 |\n|---|---:|---:|---:|---:|---:|")
    for f in ["었", "었었", "겠", "시"] + [x for x, _ in ep_tok.most_common() if x not in ("었", "었었", "겠", "시")]:
        n = len(ep_lines[f])
        R.append(f"| {f} | {ep_tok[f]} | {n} | {pct(n, N)} | {pct(n, pred_lines)} | {len(ep_eps[f])} |")
    R.append("\nkiwi が出力した EP の元の形(まとめる前): " + ", ".join(f"{k} {v}" for k, v in ep_raw.most_common()))

    # ---- PART2
    s_lines = set(li for li, _ in strict)
    b_lines = set(broad)
    R.append("\n## PART2 未来・意志・推量\n")
    R.append("| 項目 | 件数 | 行数 | 話数 | 요付き件数 |\n|---|---:|---:|---:|---:|")
    R.append(f"| -ㄹ 거(厳密: ㄹ/ETM+거・것/NNB+(이/VCP)+(EP)+語尾) | {len(strict)} | {len(s_lines)} | "
             f"{len(set(lines[i][0] for i in s_lines))} | {sum(y for _, y in strict)} |")
    R.append(f"| -ㄹ 거(広め: ㄹ/ETM+거・것/NNB すべて) | {len(broad)} | {len(b_lines)} | "
             f"{len(set(lines[i][0] for i in b_lines))} | 未実施 |")
    R.append(f"\n- 広め − 厳密 = {len(broad) - len(strict)} 件")
    R.append("\nEF としての ㄹ게・ㄹ래・ㄹ까(まとめなし。行内すべての位置/行末のみ。行末は既存の集計と比較)\n")
    R.append("| 形 | 行内すべて | 行末 | 既存の行末の値 | 一致 |\n|---|---:|---:|---:|---|")
    known = {"ㄹ게": 1015, "ㄹ까": 802, "을까": 273, "ㄹ래": 298}
    for fm in ("ㄹ게", "을게", "ㄹ래", "을래", "ㄹ까", "을까"):
        kv = known.get(fm)
        R.append(f"| {fm} | {eflist[fm]} | {final_ef[fm]} | {kv if kv is not None else '既存の出力なし'} | "
                 f"{'—' if kv is None else ('○' if kv == final_ef[fm] else '×')} |")
    R.append("\n겠 の後ろの語尾 上位20(分母: 겠 の件数 " f"{ep_tok['겠']})\n\n| 語尾/タグ | 件数 | 割合 | 요付き |\n|---|---:|---:|---:|")
    for k, v in gyet_next.most_common(20):
        R.append(f"| {k} | {v} | {pct(v, ep_tok['겠'])} | {gyet_next_yo[k]} |")

    # ---- PART3
    R.append("\n## PART3 行末の EP+EF 上位30(EF は ㅂ니다→습니다、을까→ㄹ까 でまとめる)\n")
    R.append("| EP | EF | 件数 | その EF の行末全体 | 割合 |\n|---|---|---:|---:|---:|")
    for (c, e), v in combo.most_common(30):
        R.append(f"| {c} | {e} | {v} | {final_ef_merged[e]} | {pct(v, final_ef_merged[e])} |")
    R.append(f"\n検算: 行末EFの合計 {sum(final_ef_merged.values())}(既存 50,783)、잖아 {final_ef_merged['잖아']}(既存 1,533)、"
             f"어 {final_ef_merged['어']}(既存 17,657)")

    # ---- PART4
    tot_past = sum(past_lemma.values())
    R.append(f"\n## PART4 過去形の逆引き\n\n었 の直前の用言(시 は飛ばす)の表層形 上位50(分母: 었 の前に用言がある件数 {tot_past})\n")
    R.append("| 表層 | 辞書形 | 件数 | 割合 |\n|---|---|---:|---:|")
    for (s, lm), v in past_surf.most_common(50):
        R.append(f"| {s} | {lm} | {v} | {pct(v, tot_past)} |")
    top30 = []
    if a.stems and os.path.exists(a.stems):
        with open(a.stems, encoding="utf-8-sig", newline="") as f:
            top30 = [r["語幹"] for r in csv.DictReader(f)][:30]
    tot_pred = sum(all_pred.values())
    if top30:
        cp = sum(v for lm, v in past_lemma.items() if lm in top30)
        ca = sum(v for lm, v in all_pred.items() if lm in top30)
        own30 = sum(v for _, v in all_pred.most_common(30))
        R.append(f"\n既存の逆引き表(conj_stems.csv の上位30語)によるカバー率\n")
        R.append("| 対象 | 分子 | 分母 | 割合 |\n|---|---:|---:|---:|")
        R.append(f"| 過去形(었 の前の用言) | {cp} | {tot_past} | {pct(cp, tot_past)} |")
        R.append(f"| 行内すべての位置の用言 | {ca} | {tot_pred} | {pct(ca, tot_pred)} |")
        R.append(f"| (参考)行末 어/아 の行(既存の値) | — | 17,657 | 73.8% |")
        R.append(f"| (参考)行内すべての用言の、自身の上位30語 | {own30} | {tot_pred} | {pct(own30, tot_pred)} |")
        R.append("\n既存の上位30語: " + "、".join(top30))
    else:
        R.append("\nカバー率: 未実施(--stems の conj_stems.csv が無い)")

    # ---- PART5
    rnd = random.Random(SEED)
    for name, ls, lab in (("sample_check_gyet.csv", gyet_lines, "推量／意志／定型の겠습니다／その他"),
                          ("sample_check_l_geo.csv", sorted(b_lines), "未来・予定／推量／「〜するもの」／その他"),
                          ("sample_check_eot.csv", eot_lines, "過去で正しい／解析の誤り")):
        pick = sorted(rnd.sample(ls, min(30, len(ls))))
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["話", "行番号", "行", "解析", f"判定({lab}。空欄=未検収)"])
            for li in pick:
                ep, row, t = lines[li]
                w.writerow([ep, row, t, an_cache[li], ""])
        R.append(f"\n- {name}: {len(pick)}行(母数 {len(ls)}行、判定欄は空欄)")
    return finish(a, R)


def finish(a, R):
    txt = "\n".join(R)
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
