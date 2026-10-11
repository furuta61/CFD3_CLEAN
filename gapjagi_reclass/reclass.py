import csv, sys, collections
src, out = sys.argv[1], sys.argv[2]
rs = [r for r in csv.DictReader(open(src, encoding='utf-8-sig')) if r['channel'] == '발화']
# 手判定（発話233行の通し番号）。D=①描写 R=②驚き・リアクション T=③話題転換 H=保留
D = {2,4,8,10,11,13,16,18,20,21,26,32,36,37,38,39,40,41,42,48,51,52,54,55,56,58,67,70,72,83,87,89,96,99,
     105,108,110,113,114,115,122,129,139,146,147,148,149,150,157,158,164,171,176,182,188,190,194,200,207,208,
     220,221,222,223,229,233}
T = {28,131,225}
H = {6:'断片',7:'断片',9:'断片(再生映像への語り/反応か不明)',33:'断片',35:'断片',50:'歌詞',53:'断片',68:'断片',69:'断片',
     82:'断片',86:'意味不明',97:'依頼・指示文',120:'断片',121:'断片(?なし)',130:'생각 안 나다(標識ではない)',133:'断片',
     137:'断片',141:'아 갑자기...(③の言いさしか②か不明)',143:'断片',156:'断片',160:'③の言いさしの可能性',170:'断片',
     180:'断片',183:'생각이 들다(同意の報告。標識かは不明)',187:'断片',219:'断片',232:'断片'}
HYP = {2,8,16,18,20,21,36,51,52,58,87,99,110,114,139,147,150,157,176,182,207}
lab = {}
for i in range(1, len(rs)+1):
    lab[i] = 'D' if i in D else 'T' if i in T else 'H' if i in H else 'R'
assert not (D & T) and not (D & set(H)) and not (T & set(H)) and HYP <= D
c = collections.Counter(lab.values()); n = len(rs)
name = {'D':'①描写','R':'②驚き・リアクション','T':'③話題転換','H':'保留'}
for k in 'DRTH':
    print(f"{name[k]}\t{c[k]}\t{c[k]/n*100:.1f}%")
print('計', n)
print('①のうち仮定・想定・ルール説明', len(HYP))
old = collections.Counter((r['usage_type'][0], lab[i+1]) for i, r in enumerate(rs))
print('旧→新', sorted(old.items()))
with open(out, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['no','episode','text','旧usage_type','新分類','①内訳','保留理由'])
    for i, r in enumerate(rs, 1):
        w.writerow([i, r['episode'], r['text'], r['usage_type'], name[lab[i]], '仮定・想定' if i in HYP else '', H.get(i, '')])
