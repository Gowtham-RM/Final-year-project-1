from flask import Flask, render_template, request, redirect
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 1. Load your trained DenseNet model
MODEL_PATH = 'densenet_best.keras'
model = load_model(MODEL_PATH)

# 2. Define the exact 10 classes
class_labels = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro'
]

def predict_image(image_path):
    """Processes the image and runs the DenseNet model"""
    # Read and resize image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # Expand dimensions and apply DenseNet preprocessing
    img_array = np.expand_dims(img, axis=0)
    img_processed = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_processed)
    confidence = np.max(predictions) * 100
    predicted_class = class_labels[np.argmax(predictions)]
    
    return predicted_class, confidence

# 3. Create the Web Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            predicted_class, confidence = predict_image(filepath)
            
            # Format the output clearly
            formatted_prediction = predicted_class.replace('_', ' ').title()
            
            return render_template('index.html', 
                                   uploaded_image=filepath, 
                                   prediction=formatted_prediction, 
                                   confidence=round(confidence, 2))
                                   
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)