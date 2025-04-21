import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1) Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, "dataset/3dpqc_dataset")           # contains Gray Scale/, High Threshold/, Low Threshold/
MODEL_OUT  = os.path.join(SCRIPT_DIR, "..", "models", "3dpqc_multiclass.xml")

# 2) Your class names in the order you want numeric labels
CLASS_NAMES = [
    "OK",
    "Cracks",
    "Blobs",
    "Spaghetti",
    "Stringing",
    "UnderExtrusion",
]


def load_multiclass_dataset(root_dir, subset="train"):
    """
    Walks variants under root_dir (Gray Scale / High Threshold / Low Threshold),
    then under each variant/subset dir it expects class folders named like “0.OK”, “1.Cracks”, etc.
    It splits on the first “.” to get label=int(prefix) and class_name=rest.
    """
    hog = cv2.HOGDescriptor()
    X, y = [], []
    class_names = {}  # maps label→name for later

    for variant in os.listdir(root_dir):
        var_subset = os.path.join(root_dir, variant, subset)
        if not os.path.isdir(var_subset):
            continue

        # each folder entry is e.g. "0.OK", "1.Cracks", ...
        for entry in os.listdir(var_subset):
            entry_path = os.path.join(var_subset, entry)
            if not os.path.isdir(entry_path):
                continue

            # parse label & name
            try:
                prefix, name = entry.split(".", 1)
                label = int(prefix)
            except ValueError:
                print(f"Skipping malformed folder name: {entry}")
                continue

            class_names[label] = name  # remember for inference / printing

            # load all images under that folder
            for fname in os.listdir(entry_path):
                img_path = os.path.join(entry_path, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Couldn’t read {img_path}, skipping")
                    continue

                img = cv2.resize(img, (64, 128))
                feat = hog.compute(img).flatten().astype(np.float32)
                X.append(feat)
                y.append(label)

    X = np.vstack(X)
    y = np.array(y, dtype=np.int32)

    # print summary
    counts = {lbl: int((y == lbl).sum()) for lbl in sorted(class_names)}
    print(f"Loaded {len(y)} samples for '{subset}':")
    for lbl, name in sorted(class_names.items()):
        print(f"  [{lbl}] {name}: {counts.get(lbl, 0)}")

    return X, y, class_names


def train_and_save():
    # 3) Load train & validation sets
    X_train, y_train, _ = load_multiclass_dataset(DATA_ROOT, "train")
    X_val, y_val, _ = load_multiclass_dataset(DATA_ROOT, "validate")

    # applying feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # 4) Train the multi‑class SVM
    print("\n>>> Training 6‑way SVM…")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)        # multi‑class classification
    svm.setKernel(cv2.ml.SVM_RBF)    # try RBF or POLY if you like
    svm.setC(1.0)                      # (0.01, 0.1, 1, 1.0, 10, 100)
    svm.setGamma(0.001)                 # for RBF or POLY (0.00001, 0.001, 0.01, 0.1, 1)
    #svm.setDegree(3)                  # for POLY only (2, 3, 4)
    #svm.setCoef0(1.0)                 # for POLY and Sigmoid (-1, 0, 1)
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    print("    → Training complete")

    # 5) Validate
    _, resp = svm.predict(X_val)
    preds    = resp.flatten().astype(np.int32)
    acc      = (preds == y_val).mean()
    print(f"Validation accuracy: {acc*100:.2f}%")

    # 6) Save out your final model
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    svm.save(MODEL_OUT)
    print(f"\n>>> Model saved to {MODEL_OUT}")

if __name__ == "__main__":
    train_and_save()
