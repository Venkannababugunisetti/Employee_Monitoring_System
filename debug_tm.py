# debug_tm_all.py  (you can overwrite debug_tm.py with this)

import os, glob, cv2
from classifier import TMClassifier

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

BASE_DIR = os.path.join("..", "data", "train")
CLASS_FOLDERS = {
    "working": "working",
    "not_working": "not_working",
    "person_left": "person_left",
}

print("=== DEBUG TM START ===")
print("Current working dir:", os.getcwd())

clf = TMClassifier(MODEL_PATH, LABELS_PATH)
print("Model loaded.\n")

def test_folder(true_label, folder_name):
    folder = os.path.join(BASE_DIR, folder_name)
    print(f"--- Testing folder: {folder} (true = {true_label}) ---")
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(folder, pat)))

    print("Found", len(image_paths), "images.")
    if not image_paths:
        return 0, 0

    correct = 0
    total = 0

    for img_path in image_paths[:50]:  # show first 50 for debugging
        img = cv2.imread(img_path)
        if img is None:
            print("Could not read:", img_path)
            continue

        pred = clf.predict(img).strip().lower()
        print(os.path.basename(img_path), "->", pred)

        # normalize prediction into our 3 names
        if "left" in pred:
            norm = "person_left"
        elif "not" in pred and "work" in pred:
            norm = "not_working"
        elif "work" in pred:  # and "not" not in pred
            norm = "working"
        else:
            norm = "unknown"

        if norm == true_label:
            correct += 1
        total += 1

    print(f"Correct: {correct}/{total}")
    print()
    return correct, total

tot_correct = tot_total = 0
for true_label, folder_name in CLASS_FOLDERS.items():
    c, t = test_folder(true_label, folder_name)
    tot_correct += c
    tot_total += t

print("=== OVERALL ===")
print(f"Total correct: {tot_correct}/{tot_total}")
print("Accuracy:", (tot_correct / tot_total * 100 if tot_total else 0), "%")
print("=== DEBUG TM END ===")
