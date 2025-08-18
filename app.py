
from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
with open("breast_cancer_svm_rbf.pkl", "rb") as f:
    model = pickle.load(f)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get all input values from form
        features = [
            float(request.form["clump_thickness"]),
            float(request.form["uniformity_cell_size"]),
            float(request.form["uniformity_cell_shape"]),
            float(request.form["marginal_adhesion"]),
            float(request.form["single_epithelial_cell_size"]),
            float(request.form["bare_nuclei"]),
            float(request.form["bland_chromatin"]),
            float(request.form["normal_nucleoli"]),
            float(request.form["mitoses"])
        ]

        # Convert to numpy array and reshape for model
        features_array = np.array(features).reshape(1, -1)

        # Get prediction (0 = benign, 1 = malignant)
        prediction = model.predict(features_array)[0]

        # Map prediction to label
        result = "Malignant" if prediction == 1 else "Benign"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
