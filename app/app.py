from flask import Flask, request, jsonify, render_template
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import SentimentPredictor
import pickle

app = Flask(__name__)

# Load the model and tokenizer
with open('../models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

predictor = SentimentPredictor('../models/sentiment_model.h5', tokenizer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    sentiment = predictor.predict_sentiment(review)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)