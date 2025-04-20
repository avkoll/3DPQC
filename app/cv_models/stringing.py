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

## placeholder logic to test communication
    if img.shape[1] % 2 == 0:
        result = {'defect': False, 'confidence': 0.9, 'message': "No stringing detected."}
    else:
        result = {'defect': True, 'confidence': 0.7, 'message': "Potential stringing detected."}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
