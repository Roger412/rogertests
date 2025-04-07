from glob import glob
import cv2

bad_images = []
for path in glob("data/data1/*.png"):
    img = cv2.imread(path)
    if img is None:
        bad_images.append(path)

if bad_images:
    print("❌ Corrupted or unreadable images:")
    for b in bad_images:
        print(" -", b)
else:
    print("✅ All images readable.")
