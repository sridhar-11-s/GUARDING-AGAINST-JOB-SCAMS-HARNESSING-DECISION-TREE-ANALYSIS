from flask import Flask, render_template, request
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    job_text = request.form['job_text']

    # Vectorize input text
    text_vector = vectorizer.transform([job_text])

    # Predict
    prediction = model.predict(text_vector)[0]

    # Map prediction to label
    if prediction == 1:
        result = "Fake Job"
    else:
        result = "Real Job"

    return render_template("index.html", prediction=result)

# Run app
if __name__ == "__main__":
    app.run(debug=True)
