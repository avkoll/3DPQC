import os
import json
import cv2
import numpy as np
from cv2.ml import SVM_create

# 1) Resolve paths relative to this script, no matter cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "dataset")
MODEL_OUT  = os.path.join(SCRIPT_DIR, "..", "models", "3dpqc_model.xml")

def load_coco_split(split_name):
    split_dir = os.path.join(DATA_DIR, split_name)
    print(f"\n>>> Loading split '{split_name}' from {split_dir}")
    # find JSON
    js_file = next((f for f in os.listdir(split_dir) if f.endswith('.json')), None)
    if not js_file:
        raise FileNotFoundError(f"No .json in {split_dir}")
    coco = json.load(open(os.path.join(split_dir, js_file), 'r'))

    # build maps
    id2file = {img['id']: img['file_name'] for img in coco['images']}
    anns    = {img_id: [] for img_id in id2file}
    for ann in coco['annotations']:
        anns[ann['image_id']].append(ann)

    # detect where images live
    img_folder = os.path.join(split_dir, 'images')
    if not os.path.isdir(img_folder):
        img_folder = split_dir
    print(f"    → Reading images from {img_folder}")

    # extract
    hog = cv2.HOGDescriptor()
    X, y = [], []
    for img_id, fname in id2file.items():
        img_path = os.path.join(img_folder, fname)

        # 1) Check file exists
        if not os.path.isfile(img_path):
            print(f"⚠️  File not found, skipping: {img_path}")
            continue

        # 2) Try to read
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️  Failed to load image (opencv returned None), skipping: {img_path}")
            continue

        # 3) Now it’s safe to resize
        img = cv2.resize(img, (64, 128))
        feat = hog.compute(img).flatten().astype(np.float32)
        X.append(feat)
        y.append(1 if anns[img_id] else 0)

    X = np.vstack(X)
    y = np.array(y, dtype=np.int32)
    print(f"    → Loaded {len(y)} samples: {np.sum(y==1)} defect / {np.sum(y==0)} good")
    return X, y

def train_and_save():
    # Load training data
    X_train, y_train = load_coco_split("train")

    # Initialize and train SVM
    print("\n>>> Training SVM...")
    svm = SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(0.01)
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    print("    → Training complete")

    # Ensure output dir exists
    out_dir = os.path.dirname(MODEL_OUT)
    os.makedirs(out_dir, exist_ok=True)

    # Save the model
    svm.save(MODEL_OUT)
    print(f"\n>>> Model saved to:\n    {MODEL_OUT}")

if __name__ == "__main__":
    train_and_save()
