from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

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


