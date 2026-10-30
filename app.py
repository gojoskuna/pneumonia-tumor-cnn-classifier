
import os
import uuid
import numpy as np

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

print("Starting Flask app...")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None

classes = ["Brain_Tumor", "Normal", "Pneumonia"]
from tensorflow.keras.layers import Dense

class CompatibleDense(Dense):

    @classmethod
    def from_config(cls, config):
        config = dict(config)

        quantization_config = config.pop(
            "quantization_config", None
        )

        if quantization_config is not None:
            raise ValueError(
                "This model uses quantization settings "
                "that require separate compatibility handling."
            )

        return cls(**config)
        
def get_model():
    global model

    if model is None:
        print("Loading model...")

        model = load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False,
            custom_objects={
                "Dense": CompatibleDense
            }
        )

        print("Model loaded successfully!")

    return model


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    img_path = None

    if request.method == "POST":

        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]

        if file.filename == "":
            return "Empty file", 400

        try:
            original_name = secure_filename(file.filename)

            if not original_name:
                return "Invalid filename", 400

            extension = os.path.splitext(original_name)[1].lower()

            if extension not in [".jpg", ".jpeg", ".png"]:
                return "Only JPG and PNG images are supported", 400

            filename = str(uuid.uuid4()) + extension

            filepath = os.path.join(UPLOAD_FOLDER, filename)

            file.save(filepath)

            print("Processing:", filepath)

            img = image.load_img(
                filepath,
                target_size=(224, 224)
            )

            img_array = image.img_to_array(img) / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            prediction = get_model().predict(img_array)

            predicted_class = classes[np.argmax(prediction)]

            result = predicted_class

            img_path = url_for(
                "static",
                filename=f"uploads/{filename}"
            )

        except Exception as e:
            print("ERROR:", str(e))
            return "Unable to process the image.", 500

    return render_template(
        "index.html",
        result=result,
        image=img_path
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
