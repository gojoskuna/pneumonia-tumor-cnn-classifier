import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

print("Starting Flask app...")
app = Flask(__name__)

model = None

def get_model():
    print("Loading model...")
    global model
    if model is None:
        from tensorflow.keras.models import load_model
        model = load_model("model.h5", compile=False)
    return model

classes = ["Brain_Tumor", "Normal", "Pneumonia"]

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None

    if request.method == "POST":

        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]

        if file.filename == "":
            return "Empty file"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img_path = filepath

        try:
            print("Processing:", filepath)

            img = image.load_img(filepath, target_size=(224,224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = get_model().predict(img_array)
            predicted_class = classes[np.argmax(prediction)]

            result = predicted_class

        except Exception as e:
            print("ERROR:", str(e))
            return f"Error: {str(e)}"

    return render_template("index.html", result=result, image=img_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)