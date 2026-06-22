

import Flask
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import base64

# Assuming 'model' and 'class_names' are loaded from previous cells
# If running this Flask app standalone, ensure the model and labels are loaded here:
# model = tf.keras.models.load_model('keras_model.h5', compile=False)
# with open('labels.txt', 'r') as f:
#     class_names = [line.strip() for line in f.readlines()]
# input_shape = model.input_shape
# image_size = input_shape[1]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    # Decode the base64 image data
    image_data = data['image'].split(',')[1] # Remove 'data:image/png;base64,' prefix
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Preprocess the image (same as in previous cells)
    img = img.resize((image_size, image_size))
    img_array = np.asarray(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, image_size, image_size, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make a prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index]) # Convert to float for JSON serialization

    return jsonify({
        'prediction': class_name,
        'confidence': confidence_score
    })

@app.route('/')
def home():
    return "Teachable Machine Prediction Backend is running! Send POST requests to /predict."

if __name__ == '__main__':
    # You can run this directly in your VS Code terminal:
    # python your_flask_app.py
    # Or if running from a notebook, you might need special handling for Flask.
    # For simplicity, save this code to a file (e.g., app.py) and run it.
    print("Starting Flask server...")
    app.run(debug=True, port=5000)