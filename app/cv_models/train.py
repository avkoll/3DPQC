import os, cv2, numpy as np
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import GridSearchCV
from sklearn.svm               import SVC
import joblib
import albumentations as A
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0, preprocess_input
)

# Define your augmentation pipeline once, at top
aug = A.Compose([A.Rotate(15), A.HorizontalFlip(), A.RandomBrightnessContrast(0.2,0.2)])

IMG_SIZE = 160   # bump up if you like 128→160 or 224
extractor = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

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

def img_to_embedding(cv_img):
    if cv_img.ndim == 2:                     # grayscale → RGB
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    else:                                    # BGR → RGB
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    cv_img = cv2.resize(cv_img, (IMG_SIZE, IMG_SIZE))
    x = preprocess_input(cv_img.astype("float32"))
    x = np.expand_dims(x, 0)                 # add batch dim
    emb = extractor(x, training=False).numpy().squeeze()
    return emb.astype(np.float32)            # (1280,)


def load_multiclass_dataset(root_dir, subset="train", batch_size=32):
    # 1) Gather all raw images & labels
    images, labels, class_names = [], [], {}
    for variant in os.listdir(root_dir):
        subset_dir = os.path.join(root_dir, variant, subset)
        if not os.path.isdir(subset_dir):
            continue
        for entry in os.listdir(subset_dir):
            entry_path = os.path.join(subset_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            try:
                prefix, name = entry.split(".", 1)
                label = int(prefix)
            except ValueError:
                print("Skipping malformed folder:", entry)
                continue
            class_names[label] = name
            for fname in os.listdir(entry_path):
                img = cv2.imread(os.path.join(entry_path, fname))
                if img is None:
                    print("Unreadable:", fname)
                    continue
                img = aug(image=img)["image"]
                images.append(img)
                labels.append(label)

    # 2) Batch‑process through MobileNetV2
    X_batches = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        # resize & convert to RGB
        arr = np.stack([cv2.resize(im, (IMG_SIZE, IMG_SIZE)) for im in batch])
        if arr.ndim==4 and arr.shape[-1]==3:
            arr_rgb = arr[..., ::-1]         # BGR→RGB
        else:
            arr_rgb = np.stack([cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) for im in batch])
        arr_pre = preprocess_input(arr_rgb.astype("float32"))
        embeddings = extractor(arr_pre, training=False).numpy()  # (batch_size, 1280)
        X_batches.append(embeddings)

    # 3) Assemble final X, y
    X = np.vstack(X_batches).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    # 4) Print summary
    counts = {lbl:int((y==lbl).sum()) for lbl in sorted(class_names)}
    print(f"Loaded {len(y)} samples for '{subset}':")
    for lbl,name in sorted(class_names.items()):
        print(f"  [{lbl}] {name}: {counts.get(lbl,0)}")

    return X, y, class_names


def train_and_save():
    X_train, y_train, _ = load_multiclass_dataset(DATA_ROOT, "train")
    X_val,   y_val,   _ = load_multiclass_dataset(DATA_ROOT, "validate")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": [1 / IMG_SIZE ** 2, 1e-3, 1e-2, 1e-1],  # or 1 / feature_dim
        "kernel": ["rbf"]
    }
    grid = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    svm = grid.best_estimator_
    acc = (svm.predict(X_val) == y_val).mean()
    print(f"Validation accuracy: {acc * 100:.2f}%")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(scaler, MODEL_OUT.replace(".xml", ".scaler.pkl"))
    joblib.dump(svm, MODEL_OUT.replace(".xml", ".svm.pkl"))
    print("Models saved")

if __name__ == "__main__":
    train_and_save()
