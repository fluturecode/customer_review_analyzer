import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentPredictor:
    def __init__(self, model_path, tokenizer):
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = tokenizer
        self.max_len = 200  # This should match the max_len used during training

    def predict_sentiment(self, text):
        # Preprocess the input text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        
        # Make prediction
        prediction = self.model.predict(padded)
        sentiment_class = tf.argmax(prediction, axis=1).numpy()[0]
        
        # Map sentiment class to label
        sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
        return sentiment_map[sentiment_class]

# Usage example:
if __name__ == "__main__":
    import pickle
    
    # Load the tokenizer
    with open('../models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    predictor = SentimentPredictor('../models/sentiment_model.h5', tokenizer)
    
    # Test the predictor
    test_reviews = [
        "This product exceeded my expectations!",
        "Terrible customer service, never buying again.",
        "It's an okay product, nothing special."
    ]
    
    for review in test_reviews:
        sentiment = predictor.predict_sentiment(review)
        print(f"Review: {review}")
        print(f"Sentiment: {sentiment}\n")