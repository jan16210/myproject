import Flask
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import base64
import tensorflow as tf # Import TensorFlow


app = Flask(__name__)

# --- GLOBAL VARIABLES FOR TEACHABLE MACHINE MODEL ---
# Define the path to your Teachable Machine Keras model file
MODEL_PATH = 'keras_model.h5' # <--- IMPORTANT: Update this path!
# Define the path to your labels file
LABELS_PATH = 'labels.txt' # <--- IMPORTANT: Update this path!

# Define the image size your model expects
IMAGE_SIZE = 224 # <--- IMPORTANT: Update this if your model uses a different size

# Load the Teachable Machine model and class names once when the app starts
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, 'r') as f:
        class_names = [line.strip() for line in f]
    print("Teachable Machine model and labels loaded successfully.")
except Exception as e:
    print(f"Error loading Teachable Machine model or labels: {e}")
    model = None
    class_names = []
# --- END GLOBAL VARIABLES ---


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not class_names:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        image_data = data['image'].split(',')[1] # Assumes base64 starts with 'data:image/png;base64,'
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Resize the image to the model's expected input size
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.asarray(img)

        # Normalize the image array as expected by Teachable Machine models
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        data = np.ndarray(shape=(1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = float(prediction[0][index]) # Convert to float for JSON serialization

        return jsonify({
            'prediction': class_name,
            'confidence': confidence_score
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

@app.route('/')
def home():
    return "Teachable Machine Prediction Backend is running! Send POST requests to /predict."

if __name__ == '__main__':
    print("Starting Flask server...")
    # In a production environment, you should use a WSGI server like Gunicorn
    # app.run(debug=True, host='0.0.0.0', port=5000) # Use host='0.0.0.0' to make it accessible externally
    app.run(debug=True, port=5000)
