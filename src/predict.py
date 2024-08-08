import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

class SentimentPredictor:
    def __init__(self, model_path, tokenizer_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.max_len = 200  # This should match the max_len used during training

    def predict_sentiment(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        prediction = self.model.predict(padded)
        sentiment_class = np.argmax(prediction, axis=1)[0]
        sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
        return sentiment_map[sentiment_class]

if __name__ == "__main__":
    # Adjust these paths based on your project structure
    model_path = '../models/sentiment_model.keras'
    tokenizer_path = '../models/tokenizer.pickle'
    
    predictor = SentimentPredictor(model_path, tokenizer_path)
    
    # Test the predictor
    test_reviews = [
        "This product is amazing! I love it!",
        "Terrible experience, never buying again.",
        "It's an okay product, nothing special."
    ]
    
    for review in test_reviews:
        sentiment = predictor.predict_sentiment(review)
        print(f"Review: {review}")
        print(f"Predicted sentiment: {sentiment}\n")