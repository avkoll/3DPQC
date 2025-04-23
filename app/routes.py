from flask import Blueprint, render_template, request, jsonify
import os, requests

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DETECT_URL = os.environ.get(
    'DETECT_URL',
    'http://cv_model_stringing:5001/detect'
)

@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@main.route('/analyze', methods=['POST'])
def analyze():
    img = request.files['image']
    img_path = os.path.join(UPLOAD_FOLDER, img.filename)
    img.save(img_path)

    with open(img_path, 'rb') as f:
        files = {'image': f}
        try:
            resp = requests.post(DETECT_URL, files=files, timeout=5)
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as e:
            result = {'error': 'Detection service error', 'details': str(e)}

    return render_template('index.html', result=result, filename=img.filename)
