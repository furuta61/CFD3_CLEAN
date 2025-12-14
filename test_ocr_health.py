#!/usr/bin/env python3
# ヘルスチェック: フリーズ回避パッチの動作確認
import os, cv2, time
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from vision_mobile_v12 import get_ocr, ocr_number_from_roi

print("🔍 OCR Health Check 開始...")

img = cv2.imread("IMG_5626.PNG")
assert img is not None, "IMG_5626.PNG が見つかりません"

h, w = img.shape[:2]
print(f"✅ 画像読み込み成功: {w}x{h}")

# JP225想定の上段ROIを切り出し
roi = img[int(h*0.08):int(h*0.14), int(w*0.32):int(w*0.96)]
print(f"✅ ROI切り出し: {roi.shape}")

t0 = time.perf_counter()
n = ocr_number_from_roi(roi)
dt = time.perf_counter() - t0

print(f"\n✅ OCR結果: {n}")
print(f"⏱  処理時間: {dt:.2f}秒")

if dt < 3.0:
    print("✅ 正常動作（1-3秒以内）")
else:
    print("⚠️  やや遅い（3秒超過）が、フリーズはしていません")

if n is not None:
    print("✅ 数値取得成功")
else:
    print("⚠️  数値取得失敗（ROI位置を調整してください）")
