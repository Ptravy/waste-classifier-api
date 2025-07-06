from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import numpy as np
import io
import base64
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model("sampahlaut.h5")

def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    start_time = time.time()
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Sampah Organik"
        confidence = prediction
    else:
        label = "Sampah Anorganik"
        confidence = 1 - prediction

    prediction_percentage = confidence * 100
    threshold = 60

    if prediction_percentage < threshold:
        label = "Bukan Sampah Laut"
        img_str = None
        prediction_time = round(time.time() - start_time, 2)
        return label, img_str, prediction_percentage, prediction_time

    draw = ImageDraw.Draw(img)
    color = "green" if label == "Sampah Organik" else "red"
    draw.rectangle([(10, 10), (214, 214)], outline=color, width=3)

    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    prediction_time = round(time.time() - start_time, 2)
    return label, img_str, prediction_percentage, prediction_time

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image_bytes = file.read()

        if file.content_type not in ['image/jpeg', 'image/png']:
            return jsonify({'error': 'Format gambar harus JPG/PNG.'}), 400

        label, img_str, prediction_percentage, prediction_time = predict_image(image_bytes)

        return jsonify({
            'prediction': label,
            'image': img_str,
            'prediction_percentage': f'{prediction_percentage:.2f}%',
            'prediction_time': f'{prediction_time:.2f} s'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
