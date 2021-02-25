import imghdr
import os

import PIL.Image as Image
import numpy as np
from flask import Flask, render_template, request, abort, send_from_directory
from werkzeug.utils import secure_filename

from project1.flower_model import get_class_string_from_index, reload_model, IMAGE_SIZE

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'


def validate_image(stream):
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)


@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    result = "can't decide"
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        try:
            roses = Image.open(os.path.join(app.config['UPLOAD_PATH'], filename)).resize(IMAGE_SIZE)
            roses = np.array(roses) / 255.0
            prediction_scores = reload_model.predict(np.expand_dims(roses, axis=0))
            predicted_index = np.argmax(prediction_scores)
            result = get_class_string_from_index(predicted_index)
        except ValueError:
            print("Value error and can't decide the result'")
    return render_template('index.html',
                           files=os.listdir(app.config['UPLOAD_PATH']),
                           uploaded_file=filename,
                           result=result)


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


if __name__ == '__main__':
   app.run(debug=False, host='0.0.0.0')
