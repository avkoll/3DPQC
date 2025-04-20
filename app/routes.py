from flask import Blueprint, render_template, request, jsonify
import os, requests

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DETECT_URL = os.environ.get(
    'DETECT_URL',
    'http://cv_model_stringing:5001/detect'
)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']
        img_path = os.path.join(UPLOAD_FOLDER, img.filename)
        img.save(img_path)

        with open(img_path, 'rb') as f:
            files = {'image': f}
            try:
                resp = requests.post(DETECT_URL, files=files, timeout=5)
                resp.raise_for_status()
            except requests.RequestException as e:
                return jsonify({'error': 'Detection service error', 'details': str(e)}), 502

        result = resp.json()
        return jsonify(result)

    return render_template('index.html')
