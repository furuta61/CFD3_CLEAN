from paddleocr import PaddleOCR
import cv2
import sys

ocr = PaddleOCR(lang="en", use_angle_cls=False)

img = cv2.imread(sys.argv[1])
res = ocr.ocr(img)

# 構造を完全表示
import pprint
pprint.pprint(res)
