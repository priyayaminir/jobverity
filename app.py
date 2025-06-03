import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import re

# Initialize Flask app
flask_app = Flask(__name__)

# Load model and tokenizer
model = pickle.load(open("Glove.pkl", "rb"))
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Define text preprocessing (same as used in training)
def process_text(text):
    return re.sub(r'[^a-zA-Z\s]', ' ', text).lower()

@flask_app.route("/")
def Home():
    return render_template("web.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        job_description = request.form['description']
        cleaned_text = process_text(job_description)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=50, padding='post')
        pred_prob = model.predict(padded)[0][0]
        prediction = "FAKE (Fraudulent Posting)" if pred_prob > 0.5 else "REAL (Legitimate Posting)"
        confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
        return render_template("web.html", prediction_text=f"Prediction: {prediction} ({confidence * 100:.2f}% confidence)")

if __name__ == "__main__":
    flask_app.run(debug=True)
