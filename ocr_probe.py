import sys, json
from vision_mobile_v12 import extract_watchlist_mobile_v12

if len(sys.argv) < 2:
    print("Usage: python ocr_probe.py <IMG.png>")
    sys.exit(1)

res = extract_watchlist_mobile_v12(sys.argv[1])
print(json.dumps(res, ensure_ascii=False, indent=2))
print("👉 debug: ./ocr_debug/annotated_*.png を確認してください。")
# ocr_probe.py
import sys, json, cv2
from vision_mobile_v12 import extract_watchlist_mobile_v12

if len(sys.argv) < 2:
    print("Usage: python ocr_probe.py <screenshot_path>")
    sys.exit(1)

path = sys.argv[1]
res = extract_watchlist_mobile_v12(path)
print(json.dumps(res, ensure_ascii=False, indent=2))
print("Debug images → ./ocr_debug/annotated_*.png が生成されます。")
