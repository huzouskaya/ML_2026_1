import cv2
import numpy as np
import os

img = cv2.imread("bbt.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((7,7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

rect = (600, 150, 1000, 700)
bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
cv2.grabCut(img, mask_gc := np.zeros(img.shape[:2], np.uint8), rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
mask_gc = np.where((mask_gc==2)|(mask_gc==0), 0, 255).astype(np.uint8)

cv2.imwrite("mask_morphology.png", mask)
cv2.imwrite("mask_grabcut.png", mask_gc)
cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 2)
result = cv2.bitwise_and(img, img, mask=mask_gc)
cv2.imshow("result.png", result)
cv2.waitKey(0)

iou = np.sum((mask>0) & (mask_gc>0)) / np.sum((mask>0) | (mask_gc>0))
print(f"IoU: {iou:.3f}")

def rotate(img, mask, angle):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1)
    return cv2.warpAffine(img, M, (w,h)), cv2.warpAffine(mask, M, (w,h))

def flip(img, mask, code):
    return cv2.flip(img, code), cv2.flip(mask, code)

os.makedirs("ds3/images", exist_ok=True)
os.makedirs("ds3/masks", exist_ok=True)

pairs = [(img, mask)]
for i, a in enumerate([15, -15, 30, 45, 90]): 
    img_a, mask_a = rotate(img, mask, a)
    pairs.append((img_a, mask_a))
for i, f in enumerate([0, 1, -1]): 
    img_a, mask_a = flip(img, mask, f)
    pairs.append((img_a, mask_a))

while len(pairs) < 40:
    pairs.extend(pairs[:40-len(pairs)])

for i, (im, ms) in enumerate(pairs[:40]):
    cv2.imwrite(f"ds3/images/img_{i:03d}.png", im)
    cv2.imwrite(f"ds3/masks/mask_{i:03d}.png", ms)