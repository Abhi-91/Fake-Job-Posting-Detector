from flask import Flask, render_template, request
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load saved model and tfidf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'tfidf.pkl'), 'rb') as f:
    tfidf = pickle.load(f)

# Same cleaning function from Day 2
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get text from form
    job_text = request.form['job_text']

    # Clean and vectorize
    cleaned = clean_text(job_text)
    vectorized = tfidf.transform([cleaned])

    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    if prediction == 1:
        result = "FAKE"
        confidence = round(probability[1] * 100, 2)
        color = "red"
    else:
        result = "GENUINE"
        confidence = round(probability[0] * 100, 2)
        color = "green"

    return render_template('result.html',
                           result=result,
                           confidence=confidence,
                           color=color,
                           job_text=job_text)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
