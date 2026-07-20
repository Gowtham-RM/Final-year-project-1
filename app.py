from flask import Flask, render_template, request, redirect
import os
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------
# Load EfficientNet Model
# --------------------------------------------------

MODEL_PATH = "efficientnet_phase2_best.keras"

model = load_model(MODEL_PATH)

# --------------------------------------------------
# Class Labels
# --------------------------------------------------

class_labels = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro"
]

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_image(image_path):

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))

    image = np.array(image, dtype=np.float32)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    prediction = model.predict(image, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))*100

    predicted_class = class_labels[predicted_index]

    return predicted_class, confidence


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/", methods=["GET","POST"])

def index():

    if request.method=="POST":

        if "file" not in request.files:
            return redirect(request.url)

        file=request.files["file"]

        if file.filename=="":
            return redirect(request.url)

        filepath=os.path.join(app.config["UPLOAD_FOLDER"],file.filename)

        file.save(filepath)

        prediction,confidence=predict_image(filepath)

        prediction=prediction.replace("_"," ").title()

        return render_template(
            "index.html",
            uploaded_image=filepath,
            prediction=prediction,
            confidence=round(confidence,2)
        )

    return render_template("index.html")


if __name__=="__main__":
    app.run(debug=True)