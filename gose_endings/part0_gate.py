#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GOSE 語尾・助詞探索 第1便 PART0 ゲート

(i)  過去方式の再現: scene_extraction.py と同じ入力(*_utterances.json)・同じ数え方
     (EFタグ かつ form 完全一致)で 거든/잖아 を数え、concentration_ranking.csv と比較する。
(ii) 今回の母集団: corpus.csv からテロップ行を除いた全行(重複集約なし)で同じ数え方をし、
     (i) とのずれ、過去KWIC(geodeun_kwic_gose.csv / janha_kwic_gose.csv)とのずれを出す。
あわせて vocab_coverage_v3_raw.pkl に E系・J系タグがあるかを調べる。

使い方(例):
  python3 part0_gate.py \
    --utt-dir "~/Desktop/clode code/seventeen_analysis" \
    --corpus corpus.csv --pkl vocab_coverage_v3_raw.pkl \
    --concentration concentration_ranking.csv \
    --kwic-geodeun geodeun_kwic_gose.csv --kwic-janha janha_kwic_gose.csv \
    --out out_part0
引数はどれも省略可。省略したものはスキップして報告に「未実行」と書く。
"""
import argparse
import csv
import glob
import io
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import kiwipiepy
from kiwipiepy import Kiwi

TARGETS = ("거든", "잖아")
TOL = 0.05
HANGUL = re.compile(r"[가-힣]")
BRACKET = re.compile(r"\[[^\]]*\]")
TELOP_TAG = "0_対象外_텔롭"

kiwi = Kiwi()


def clean_text(t):
    # kpop_analyze.clean_text は消失。空白の正規化のみ行う(差は total_tokens の比較で見る)
    return re.sub(r"\s+", " ", t).strip()


def ef_counts(toks):
    """EFトークンを form 別に数える。거든/거든요 を区別したまま返す"""
    c = Counter()
    for t in toks:
        if t.tag == "EF":
            for w in TARGETS:
                if t.form == w:
                    c[(w, "exact")] += 1
                elif t.form == w + "요":
                    c[(w, "yo")] += 1
    return c


def pct_diff(new, old):
    if old == 0:
        return None if new == 0 else float("inf")
    return (new - old) / old


def fmt_pct(x):
    if x is None:
        return "—"
    if x == float("inf"):
        return "inf"
    return f"{x * 100:+.1f}%"


# ---------------------------------------------------------------- (i)
def load_json_utterances(path):
    data = json.load(open(path, encoding="utf-8"))
    segs = data if isinstance(data, list) else data.get("segments", data)
    out = []
    for s in segs:
        t = str(s.get("text", "")).strip() if isinstance(s, dict) else str(s).strip()
        if t:
            cl = clean_text(t)
            if HANGUL.search(cl):
                out.append(cl)
    return out


def run_i(utt_dir):
    files = sorted(glob.glob(os.path.join(os.path.expanduser(utt_dir), "*_utterances.json")))
    ep = {}
    for f in files:
        ep_id = os.path.splitext(os.path.basename(f))[0]
        tot = 0
        c = Counter()
        for u in load_json_utterances(f):
            toks = kiwi.tokenize(u)
            tot += len(toks)
            c += ef_counts(toks)
        ep[ep_id] = {"total_tokens": tot, "counts": c}
    return files, ep


def load_concentration(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                if rows:
                    break  # 代表KWICの追記部分に入ったら終了
                continue
            rows.append(line)
    rd = csv.DictReader(io.StringIO("".join(rows)))
    return [r for r in rd if r.get("corpus") == "GOSE"]


# ---------------------------------------------------------------- (ii)
def is_telop(row, text):
    for v in row.values():
        if v and TELOP_TAG in str(v):
            return True, "tag"
    # タグ列が無い場合の暫定規則: [..] を除くと何も残らない行
    if BRACKET.search(text) and not BRACKET.sub("", text).strip():
        return True, "heuristic"
    return False, None


def run_ii(corpus_path):
    stats = Counter()
    eps = set()
    ep_counts = defaultdict(Counter)
    ep_tokens = Counter()
    line_final = Counter()
    with open(corpus_path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        for row in rd:
            stats["rows"] += 1
            text = (row.get("text") or "").strip()
            ep_id = row.get("episode") or row.get("video_id") or ""
            eps.add(ep_id)
            tel, how = is_telop(row, text)
            if tel:
                stats[f"telop_{how}"] += 1
                continue
            if BRACKET.search(text):
                stats["partial_bracket_kept"] += 1
            if not HANGUL.search(text):
                stats["no_hangul"] += 1
                continue
            stats["analyzed"] += 1
            toks = kiwi.tokenize(clean_text(text))
            ep_tokens[ep_id] += len(toks)
            ep_counts[ep_id] += ef_counts(toks)
            # 行末(記号を除いた最後の形態素)の 거든/잖아
            core = [t for t in toks if not t.tag.startswith("S")]
            if core and core[-1].tag == "EF":
                for w in TARGETS:
                    if core[-1].form in (w, w + "요"):
                        line_final[(w, "exact" if core[-1].form == w else "yo")] += 1
    return cols, stats, eps, ep_counts, ep_tokens, line_final


def kwic_summary(path):
    by = Counter()
    n = 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            by[(r.get("surface", ""), r.get("kiwi_tag", ""), r.get("position", ""))] += 1
    return n, by


# ---------------------------------------------------------------- pkl
def inspect_pkl(path):
    try:
        obj = pickle.load(open(path, "rb"))
    except Exception as e:  # pandas 等が必要な場合もある
        return {"error": repr(e)}
    tags = Counter()
    seen = [0]

    def walk(o, depth=0):
        if seen[0] > 5_000_000 or depth > 8:
            return
        seen[0] += 1
        tag = getattr(o, "tag", None)
        if isinstance(tag, str):
            tags[tag] += 1
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("tag", "pos") and isinstance(v, str):
                    tags[v] += 1
                else:
                    walk(k, depth + 1)
                    walk(v, depth + 1)
        elif isinstance(o, (list, tuple, set)):
            if isinstance(o, tuple) and len(o) >= 2 and all(isinstance(x, str) for x in o[:2]) \
                    and re.fullmatch(r"[A-Z]{1,3}(-[A-Z])?", o[1]):
                tags[o[1]] += 1
                return
            for x in o:
                walk(x, depth + 1)
        elif hasattr(o, "to_dict"):  # pandas
            walk(o.to_dict(), depth + 1)

    walk(obj)
    top = type(obj).__name__
    shape = len(obj) if hasattr(obj, "__len__") else None
    keys = list(obj.keys())[:20] if isinstance(obj, dict) else None
    return {"type": top, "len": shape, "keys": keys, "tags": tags}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utt-dir")
    ap.add_argument("--corpus")
    ap.add_argument("--pkl")
    ap.add_argument("--concentration")
    ap.add_argument("--kwic-geodeun")
    ap.add_argument("--kwic-janha")
    ap.add_argument("--out", default="out_part0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = [f"# PART0 ゲート報告\n\nkiwipiepy {kiwipiepy.__version__}\n"]
    gate_ok = True

    # pkl
    R.append("## pkl のタグ\n")
    if a.pkl:
        p = inspect_pkl(a.pkl)
        if "error" in p:
            R.append(f"読み込み失敗: {p['error']}\n")
        else:
            t = p["tags"]
            e = {k: v for k, v in t.items() if k.startswith("E")}
            j = {k: v for k, v in t.items() if k.startswith("J")}
            R.append(f"- 型: {p['type']} / 長さ: {p['len']} / キー先頭: {p['keys']}")
            R.append(f"- 検出タグ総数: {sum(t.values())} 種類: {len(t)}")
            R.append(f"- E系: {e or 'なし'}")
            R.append(f"- J系: {j or 'なし'}")
            R.append(f"- 全タグ上位: {t.most_common(40)}\n")
    else:
        R.append("未実行\n")

    # (i)
    R.append("## (i) 過去方式の再現(utterances.json, EF×form完全一致)\n")
    ep_i = {}
    if a.utt_dir:
        files, ep_i = run_i(a.utt_dir)
        tot_tok = sum(v["total_tokens"] for v in ep_i.values())
        agg = Counter()
        for v in ep_i.values():
            agg += v["counts"]
        R.append(f"- ファイル数: {len(files)}(過去ログ: 123)")
        R.append(f"- 総形態素: {tot_tok}")
        for w in TARGETS:
            R.append(f"- {w}: 完全一致(過去方式) {agg[(w, 'exact')]} / {w}요(EF一語) {agg[(w, 'yo')]}")
        if a.concentration:
            R.append("\n| 語 | 話 | 過去count | 再現count | 差 | 過去tokens | 再現tokens | 差 |")
            R.append("|---|---|---:|---:|---:|---:|---:|---:|")
            sum_old = Counter()
            sum_new = Counter()
            rows_out = []
            for r in load_concentration(a.concentration):
                w, e = r["word"], r["episode"]
                old_c, old_t = int(r["count"]), int(r["total_tokens"])
                v = ep_i.get(e)
                new_c = v["counts"][(w, "exact")] if v else None
                new_t = v["total_tokens"] if v else None
                dc = pct_diff(new_c, old_c) if v else None
                dt = pct_diff(new_t, old_t) if v else None
                R.append(f"| {w} | {e} | {old_c} | {new_c} | {fmt_pct(dc)} | {old_t} | {new_t} | {fmt_pct(dt)} |")
                rows_out.append([w, e, old_c, new_c, old_t, new_t])
                if v:
                    sum_old[w] += old_c
                    sum_new[w] += new_c
                else:
                    gate_ok = False
            for w in TARGETS:
                d = pct_diff(sum_new[w], sum_old[w])
                ok = d is not None and abs(d) <= TOL
                gate_ok &= ok
                R.append(f"\n- {w} 上位5話合計: 過去 {sum_old[w]} / 再現 {sum_new[w]} / {fmt_pct(d)} → {'±5%以内' if ok else '±5%超'}")
            with open(os.path.join(a.out, "part0_episode_compare.csv"), "w", newline="", encoding="utf-8-sig") as f:
                w_ = csv.writer(f)
                w_.writerow(["word", "episode", "old_count", "new_count", "old_tokens", "new_tokens"])
                w_.writerows(rows_out)
        else:
            gate_ok = False
            R.append("- concentration_ranking.csv 未指定のため比較なし")
    else:
        gate_ok = False
        R.append("未実行\n")

    # (ii)
    R.append("\n## (ii) corpus.csv・テロップ除外・集約なし\n")
    if a.corpus:
        cols, st, eps, ep_c, ep_t, lf = run_ii(a.corpus)
        agg = Counter()
        for c in ep_c.values():
            agg += c
        R.append(f"- 列: {cols}")
        R.append(f"- 総行数: {st['rows']} / 話数: {len(eps)}")
        R.append(f"- テロップ除外: タグ一致 {st['telop_tag']} / 暫定規則([..]のみの行) {st['telop_heuristic']}")
        R.append(f"- 除外後の行数: {st['rows'] - st['telop_tag'] - st['telop_heuristic']}"
                 f"(うちハングル無しで解析対象外 {st['no_hangul']}、解析 {st['analyzed']})")
        R.append(f"- [..] を一部含むが残した行: {st['partial_bracket_kept']}")
        R.append(f"- 総形態素: {sum(ep_t.values())}")
        for w in TARGETS:
            R.append(f"- {w}: 完全一致 {agg[(w, 'exact')]} / {w}요 {agg[(w, 'yo')]}"
                     f" / うち行末 {lf[(w, 'exact')]} + {lf[(w, 'yo')]}")
        if ep_i:
            for w in TARGETS:
                i_n = sum(v["counts"][(w, "exact")] for v in ep_i.values())
                R.append(f"- (ii)-(i) {w} 完全一致: {agg[(w, 'exact')]} vs {i_n} ({fmt_pct(pct_diff(agg[(w, 'exact')], i_n))})")
        for w, path in (("거든", a.kwic_geodeun), ("잖아", a.kwic_janha)):
            if path:
                n, by = kwic_summary(path)
                R.append(f"\n- 過去KWIC {os.path.basename(path)}: {n}行")
                for k, v in by.most_common(10):
                    R.append(f"  - surface/tag/position={k}: {v}")
                mine = agg[(w, "exact")] + agg[(w, "yo")]
                R.append(f"  - 比較: 今回 EF {w}+{w}요 = {mine} vs KWIC {n} ({fmt_pct(pct_diff(mine, n))})")
    else:
        R.append("未実行\n")

    R.append(f"\n## ゲート判定(i 上位5話合計 ±5%): {'通過' if gate_ok else '不通過または未判定'}\n")
    R.append("注: 暫定テロップ規則は [..] だけで構成される行のみ。正式な 0_対象外_텔롭 タグ列があればそれを優先。")
    txt = "\n".join(R)
    open(os.path.join(a.out, "part0_report.md"), "w", encoding="utf-8").write(txt)
    print(txt)
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
