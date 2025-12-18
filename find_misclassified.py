# save as src/find_misclassified.py
import os
import cv2
from classifier import TMClassifier

MODEL = "best_model.h5"   # or keras_model.h5 if TM export
LABELS = "labels.txt"
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLASS_FOLDERS = {
    "working": os.path.join(BASE, "data", "train", "working"),
    "not_working": os.path.join(BASE, "data", "train", "not_working"),
    "person_left": os.path.join(BASE, "data", "train", "person_left"),
}

clf = TMClassifier(MODEL, LABELS)

for true_label, folder in CLASS_FOLDERS.items():
    if not os.path.isdir(folder):
        print("Missing:", folder)
        continue
    files = sorted(os.listdir(folder))
    print("Testing folder:", folder, "count:", len(files))
    mistakes = []
    for f in files[:200]:   # limit first 200 for quick check
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        if img is None:
            continue
        pred = clf.predict(img).strip().lower()
        if pred.replace(" ", "_") != true_label:
            mistakes.append((f, pred))
    print("Misclassified (first 50):", mistakes[:50])
    print()
