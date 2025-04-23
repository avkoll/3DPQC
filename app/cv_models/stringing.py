import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input

# 1) Load your scaler and your SVM (sklearn) instead of cv2
MODEL_DIR    = os.environ.get("MODEL_PATH", "../models")
SCALER_PATH  = os.path.join(MODEL_DIR, "3dpqc_multiclass.scaler.pkl")
SVM_PATH     = os.path.join(MODEL_DIR, "3dpqc_multiclass.svm.pkl")
scaler       = joblib.load(SCALER_PATH)
svm          = joblib.load(SVM_PATH)

# 2) Re‑instantiate the same CNN extractor you trained with:
IMG_SIZE = 224
extractor = EfficientNetB1(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

CLASS_NAMES = [
    "OK",
    "Cracks",
    "Blobs",
    "Spaghetti",
    "Stringing",
    "UnderExtrusion",
]

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error':'No image received'}), 400
    file_bytes = request.files['image'].read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error':'Invalid image'}), 400

    # 3) Preprocess exactly as in training (no HOG)
    #    - Resize to 224×224, BGR→RGB, preprocess_input
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    x           = preprocess_input(img_rgb.astype('float32'))
    x           = np.expand_dims(x, 0)   # batch dim
    emb         = extractor(x, training=False).numpy().squeeze()  # (1280,)

    # 4) Scale + predict
    feat        = scaler.transform(emb.reshape(1, -1))
    probabilities = svm.predict_proba(feat)[0]
    label_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[label_idx])
    label_name = CLASS_NAMES[label_idx]

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_classes = [
        {"class_id": int(i), "class_name": CLASS_NAMES[i], "confidence": float(probabilities[i])}
        for i in top_indices
    ]

    return jsonify({
        'class_id': label_idx,
        'class_name': label_name,
        'confidence': confidence,
        'message': f"Detected: {label_name}",
        'top_predictions': top_classes
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
