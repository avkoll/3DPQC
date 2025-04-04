from flask import Blueprint, render_template, request, jsonify
import os

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']
        img_path = os.path.join(UPLOAD_FOLDER, img.filename)
        img.save(img_path)

        # Placeholder: image processing logic goes here
        result = {"defects": []}

        return jsonify(result)

    return render_template('index.html')
