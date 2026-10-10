#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用言と語尾のつなぎ目の変わり方(活用の種類)の頻度 — GOSE / Run BTS / BOMB

前処理V2・kiwipiepy 0.23.2。既存の出力は上書きしない(既定 out_conj_freq)。
使い方(コーパス分析フォルダで):
  python3 conj_freq.py --gose seventeen_analysis/output/ani_textbook_pipeline/corpus.csv \
      --bts bts_pipeline/bts_textbook_pipeline/bts_corpus.csv --out out_conj_freq
"""
import argparse, csv, os, random, re, sys
from collections import Counter, defaultdict
import kiwipiepy
from kiwipiepy import Kiwi

BR = re.compile(r"\[[^\]]*\]"); PA = re.compile(r"\([^)]*\)"); H = re.compile(r"[가-힣]")
def clean(t): return re.sub(r"\s+", " ", PA.sub(" ", BR.sub(" ", t))).strip()
bt = lambda t: t.split("-")[0]
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"; JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
def dec(ch):
    o = ord(ch) - 0xAC00
    return (JUNG[o % 588 // 28], JONG[o % 28]) if 0 <= o < 11172 else None
JM = {"ᆫ": "ㄴ", "ᆯ": "ㄹ", "ᆸ": "ㅂ", "ᆷ": "ㅁ", "ᆻ": "ㅆ"}
PRED = {"VV", "VA", "VX", "XSV", "XSA", "VCP", "VCN"}
REG_REU = ("따르", "치르", "들르", "잇따르")      # 르で終わるが規則的な으の脱落
KEEP = "形が変わらない(そのまま／으・아/어が付くだけ)"
ORDER = [KEEP, "하다→해", "이다・아니다", "母音縮約(ㅗ・ㅜ→ㅘ・ㅝ)", "母音縮約(ㅏ・ㅓ:同じ母音が重なる)",
         "母音縮約(ㅣ→ㅕ)", "母音縮約(ㅚ→ㅙ)", "母音縮約(ㅐ・ㅔなど)", "ㅂ不規則", "ㅎ不規則", "ㄹ語幹の脱落",
         "르不規則", "으の脱落", "ㄷ不規則", "ㅅ不規則", "その他の不規則"]
EXPECT_GOSE = {"合計": 157595, KEEP: 107617, "하다→해": 10175, "이다・아니다": 8701, "母音縮約(ㅗ・ㅜ→ㅘ・ㅝ)": 7484,
               "ㅂ不規則": 1614, "ㄹ語幹の脱落": 1113, "르不規則": 1035}

def classify(p, e):
    tg = bt(p.tag); ef = e.form; f0 = JM.get(ef[0], ef[0]); stem = p.form
    last = dec(stem[-1]) if stem else None
    eo = ef.startswith(("어", "었"))
    if tg in ("VCP", "VCN"):
        return "이다・아니다" if (eo or ef[0] in "야에") else KEEP
    if stem == "하" and eo:
        return "하다→해"
    if p.tag.endswith("-I") and last:
        if not (eo or f0 in "으은을읍ㄴㄹㅂ" or ef[0] in "니네"):
            return KEEP
        j = last[1]
        if stem.endswith("르"): return "르不規則"
        return {"ㅂ": "ㅂ不規則", "ㄷ": "ㄷ不規則", "ㅅ": "ㅅ不規則", "ㅎ": "ㅎ不規則"}.get(j, "その他の不規則")
    if last and last[1] == "ㄹ" and (f0 in "ㄴㅂㅅ" or ef[0] in "니네나세시습ᆫᆯᆸ" or ef.startswith(("으", "은", "을", "읍", "는"))):
        return "ㄹ語幹の脱落"
    if last and last[1] == " " and eo:
        v = last[0]
        if v == "ㅡ": return "르不規則" if stem.endswith("르") and stem not in REG_REU else "으の脱落"
        if v in "ㅏㅓ": return "母音縮約(ㅏ・ㅓ:同じ母音が重なる)"
        if v in "ㅗㅜ": return "母音縮約(ㅗ・ㅜ→ㅘ・ㅝ)"
        if v == "ㅣ": return "母音縮約(ㅣ→ㅕ)"
        if v == "ㅚ": return "母音縮約(ㅚ→ㅙ)"
        return "母音縮約(ㅐ・ㅔなど)"
    return KEEP

def run(kiwi, items):
    L = [(ep, row, clean(t)) for ep, row, t in items]; L = [x for x in L if H.search(x[2])]
    C = Counter(); lem = defaultdict(Counter); eps = defaultdict(set); ex = defaultdict(list); tot = 0
    for (ep, row, t), ts in zip(L, kiwi.tokenize([x[2] for x in L])):
        for i, p in enumerate(ts[:-1]):
            e = ts[i + 1]
            if bt(p.tag) not in PRED or not bt(e.tag).startswith("E"):
                continue
            k = classify(p, e); tot += 1
            C[k] += 1; lem[k][p.form + "다"] += 1; eps[k].add(ep)
            ex[k].append((ep, row, t, f"{p.form}/{p.tag}+{e.form}/{e.tag}"))
    return dict(lines=len(L), tot=tot, C=C, lem=lem, eps=eps, ex=ex)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gose", required=True); ap.add_argument("--bts", required=True)
    ap.add_argument("--out", default="out_conj_freq"); a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"{a.out} に既にファイルがあります。上書きしないため停止します。")
    os.makedirs(a.out, exist_ok=True)
    kiwi = Kiwi(); O = [f"# 活用の種類の頻度\n\nkiwipiepy {kiwipiepy.__version__} / 前処理V2\n"]
    with open(a.gose, encoding="utf-8-sig", newline="") as f:
        g = [(r.get("episode", ""), i, r.get("text") or "") for i, r in enumerate(csv.DictReader(f), 1)]
    with open(a.bts, encoding="utf-8-sig", newline="") as f:
        b = list(csv.DictReader(f))
    R = {"GOSE": run(kiwi, g)}
    for s in ("run", "bomb"):
        R[s] = run(kiwi, [(r["clip_id"], r.get("line_no", ""), r.get("text") or "") for r in b if r["source"] == s])
    G = R["GOSE"]
    O.append("## PART0 再現確認(GOSE)\n\n| 項目 | 期待値 | 今回 | 一致 |\n|---|---:|---:|---|")
    ok = True
    for k, v in EXPECT_GOSE.items():
        got = G["tot"] if k == "合計" else G["C"][k]
        O.append(f"| {k} | {v} | {got} | {'○' if got == v else '×'} |"); ok &= got == v
    if not ok:
        O.append("\n**不一致のため以降は参考値。PART1以降の数字を使う前に原因を報告すること。**")
    O.append("\n## PART1 種類ごとの件数と割合(分母:用言と語尾のつなぎ目の数)\n")
    O.append("| 種類 | GOSE | GOSE% | Run BTS | Run% | BOMB | BOMB% |\n|---|---:|---:|---:|---:|---:|---:|")
    for k in ORDER:
        O.append(f"| {k} | {G['C'][k]} | {G['C'][k]/G['tot']*100:.1f}% | {R['run']['C'][k]} | {R['run']['C'][k]/R['run']['tot']*100:.1f}% | "
                 f"{R['bomb']['C'][k]} | {R['bomb']['C'][k]/R['bomb']['tot']*100:.1f}% |")
    O.append(f"| 合計 | {G['tot']} | | {R['run']['tot']} | | {R['bomb']['tot']} | |")
    O.append("\n## PART2 種類ごとの語の偏り(上位1語・上位3語が占める割合、上位8語)\n")
    O.append("| 種類 | コーパス | 件数 | 上位1語の割合 | 上位3語の割合 | 上位8語 | 出現話数 |\n|---|---|---:|---:|---:|---|---:|")
    for k in ORDER[1:]:
        for n in ("GOSE", "run", "bomb"):
            X = R[n]; c = X["C"][k]
            if not c: O.append(f"| {k} | {n} | 0 | — | — | — | 0 |"); continue
            mc = X["lem"][k].most_common(8)
            t1 = mc[0][1] / c * 100; t3 = sum(v for _, v in mc[:3]) / c * 100
            O.append(f"| {k} | {n} | {c} | {t1:.0f}% | {t3:.0f}% | {'、'.join(f'{w}{v}' for w, v in mc)} | {len(X['eps'][k])} |")
    rnd = random.Random(42); rows = []
    for k in ORDER[1:]:
        for n in ("GOSE", "run"):
            exs = R[n]["ex"][k]
            for ep, row, t, an in rnd.sample(exs, min(10, len(exs))):
                rows.append([k, n, ep, row, t, an, ""])
    with open(os.path.join(a.out, "sample_check_conj.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["種類", "コーパス", "話", "行番号", "行", "解析", "判定(正しい/誤り。空欄=未検収)"]); w.writerows(rows)
    O.append(f"\n## PART3 検収用サンプル\n\n- sample_check_conj.csv: {len(rows)}行(種類×コーパス(GOSE・Run BTS)ごとに最大10行、シード42、判定欄は空欄)")
    txt = "\n".join(O)
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write(txt); print(txt)

if __name__ == "__main__":
    main()
