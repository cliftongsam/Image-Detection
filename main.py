from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load the model
model = load_model('best_model.h5')  # Replace with your actual .h5 model file path
print("Model loaded successfully!")

# Define class names (manually if not saved in the .h5 file)
class_names = ['Buildings', 'Forests', 'Glaciers', 'Mountains', 'Seas', 'Streets']

app = Flask(__name__)


def predict_image(img_path):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(256, 256))  # Adjust target size to your model's input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]  # Get class with highest probability
    return predicted_class


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        # Save the uploaded file to the 'static' directory
        file_path = os.path.join("static", file.filename)
        file.save(file_path)

        # Predict the class of the uploaded image
        result = predict_image(file_path)

        # Render the HTML page with the result
        return render_template("index.html", image=file_path, result=result)

    # Render the default HTML page
    return render_template("index.html", image=None, result=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
