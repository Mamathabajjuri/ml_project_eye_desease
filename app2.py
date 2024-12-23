from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = "C:\\Users\\S.JHANSY\\OneDrive\\Desktop\\EYE2\\EYE2\\model1\\full_model1.h5"
model = load_model(model_path)

# Define image preprocessing parameters
img_height, img_width = 150, 150

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img_path = os.path.join('uploads', image.filename)
    image.save(img_path)
    
    # Load and preprocess the image
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Define class labels (adjust based on your model)
    class_labels = ['normal', 'Diabatics', 'Glocoma', 'Catarate', 'Age related', 'Highpertention', 'pathalogical', 'abnormal']
    predicted_label = class_labels[predicted_class]
    
    return render_template('prediction.html', prediction=predicted_label, image_path=img_path)
    

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True,host="0.0.0.0",port=5000)
