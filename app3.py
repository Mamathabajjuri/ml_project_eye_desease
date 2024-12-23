from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import mysql.connector

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for session management

# MySQL configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="user_db"
)

# Load the trained model
model_path = "C:\\Users\\S.JHANSY\\OneDrive\\Desktop\\EYE2\\EYE2\\model1\\full_model1.h5"
model = load_model(model_path)

# Define image preprocessing parameters
img_height, img_width = 150, 150

@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html')  # Render index page if logged in
    return redirect(url_for('login'))  # Redirect to login page by default

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = db.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            db.commit()
            return render_template('login.html', message="Registration successful! You can now log in.")
        except mysql.connector.IntegrityError:
            return render_template('register.html', message="Username already exists.")
        finally:
            cursor.close()

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = db.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()

        if result and result[0] == password:  # Check plain password
            session['username'] = username  # Store username in session
            return redirect(url_for('index'))  # Redirect to index after login
        else:
            return render_template('login.html', message="Invalid credentials")

    return render_template('login.html')

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
    class_labels = ['normal', 'Diabetics', 'Glaucoma', 'Cataract', 'Age-related', 'Hypertension', 'pathological', 'abnormal']
    predicted_label = class_labels[predicted_class]

    return render_template('prediction.html', prediction=predicted_label, image_path=img_path)

@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove username from session
    return redirect(url_for('login'))  # Redirect to login page

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, host="0.0.0.0", port=5000)
