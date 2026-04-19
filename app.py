import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model("model.h5")


classes = ["Brain_Tumor", "Normal", "Pneumonia"]

UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            img_path = filepath

            img = image.load_img(filepath, target_size=(224,224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = classes[np.argmax(prediction)]

            result = predicted_class

    return render_template("index.html", result=result, image=img_path)

if __name__ == "__main__":
    app.run(debug=True)