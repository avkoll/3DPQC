import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "../models/3dpqc_model.xml")
svm        = cv2.ml.SVM_load(MODEL_PATH)
hog        = cv2.HOGDescriptor()

@app.route('/detect', methods=['POST'])

def detect_stringing():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file received'}), 400

    file = request.files['image']
    file_bytes = file.read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    try:
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('Not image file')
    except Exception as e:
        return jsonify({'error': str(e)}), 400


    # convert to grayscale + resize exactly as in training
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 128))
    feat = hog.compute(small).flatten().astype(np.float32)
    _, resp = svm.predict(feat.reshape(1, -1))
    is_def = bool(int(resp[0, 0]))

    # (optional) you can set confidence=1.0 for now or implement raw‑score extraction later
    result = {
        'defect': is_def,
        'confidence': 1.0,
        'message': "Potential stringing detected." if is_def else "No stringing detected."
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
