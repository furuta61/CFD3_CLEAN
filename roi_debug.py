import cv2
import sys

# 画像読み込み
img = cv2.imread(sys.argv[1])
h, w, _ = img.shape

print(f"画像サイズ: {w} x {h}")

# スケール計算
scale_y = h / 2868.0
scale_x = w / 1320.0

print(f"スケール: x={scale_x:.3f}, y={scale_y:.3f}")

# ROI抽出テスト
ROI_MAP = {
    "JP225":  (430,  520, 1320-430, 200),
    "NAS100": (430,  870, 1320-430, 200),
    "GER40":  (430, 1210, 1320-430, 200),
    "XAUUSD": (430, 1550, 1320-430, 200),
}

for label, (x, y, rw, rh) in ROI_MAP.items():
    X = int(x * scale_x)
    Y = int(y * scale_y)
    W = int(rw * scale_x)
    H = int(rh * scale_y)
    
    print(f"\n{label}:")
    print(f"  元座標: ({x}, {y}, {rw}, {rh})")
    print(f"  実座標: ({X}, {Y}, {W}, {H})")
    print(f"  範囲: x={X}~{X+W}, y={Y}~{Y+H}")
    
    # ROI抽出して保存
    roi = img[Y:Y+H, X:X+W]
    cv2.imwrite(f"roi_{label}.png", roi)
    print(f"  保存: roi_{label}.png")

print("\nROI画像を保存しました。目視で数字が含まれているか確認してください。")
