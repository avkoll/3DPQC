import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# 1) Point at your new multiclass model
MODEL_PATH = os.environ.get("MODEL_PATH", "../models/3dpqc_multiclass.xml")
svm        = cv2.ml.SVM_load(MODEL_PATH)
hog        = cv2.HOGDescriptor()

# 2) Class names in the exact order used during training
CLASS_NAMES = [
    "OK",
    "Cracks",
    "Blobs",
    "Spaghetti",
    "Stringing",
    "UnderExtrusion",
]

@app.route('/detect', methods=['POST'])
def detect():
    # 3) Basic upload boilerplate
    if 'image' not in request.files:
        return jsonify({'error': 'No image file received'}), 400

    file_bytes = request.files['image'].read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    # 4) Preprocess exactly as in training
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64,128))
    feat  = hog.compute(small).flatten().astype(np.float32)

    # 5) Predict — returns the index of the winning class
    _, resp    = svm.predict(feat.reshape(1, -1))
    label_idx  = int(resp[0,0])
    label_name = CLASS_NAMES[label_idx]

    print(f"Raw SVM response matrix: {resp}, selected label: {label_idx} ('{label_name}')")

    # 6) Return JSON
    return jsonify({
        'class_id':   label_idx,
        'class_name': label_name,
        'message':    f"Detected: {label_name}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
