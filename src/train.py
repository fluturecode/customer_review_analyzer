# src/train.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pickle

from data_preprocessing import load_data, preprocess_data
from model import build_model, train_model, evaluate_model

if __name__ == "__main__":
    # Load data
    data = load_data('data/raw_data.csv')
    print("Data loaded successfully. Shape:", data.shape)

    # Print sentiment distribution
    print("Sentiment distribution in training data:")
    print(data['sentiment'].value_counts(normalize=True))

    # Preprocess data
    X, tokenizer = preprocess_data(data)
    
    # Encode sentiments
    le = LabelEncoder()
    y = le.fit_transform(data['sentiment'])
    
    # Debug: Print sample of preprocessed data
    print("Sample of preprocessed data:")
    print(X[:5])
    print("Sample of encoded sentiments:")
    print(y[:5])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Debug: Print shapes after split
    print("Train-Test Split:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Debug: Print class weights
    print("Class weights:")
    print(class_weight_dict)
    
    # Build model
    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(vocab_size)
    
    # Print model summary
    print("Model Summary:")
    model.summary()
    
    # Train model
    print("Starting model training...")
    history = train_model(model, X_train, y_train, X_test, y_test, class_weight_dict)
    print("Model training completed.")
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = evaluate_model(model, X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Calculate additional metrics
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("Plot saved as 'training_history.png'")
    
    # Save model and tokenizer
    model.save('../models/sentiment_model.keras')
    with open('../models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Model and tokenizer saved successfully.")
    
    # Test on sample reviews
    sample_reviews = [
        "This product is amazing! I love it!",
        "Terrible experience, never buying again.",
        "It's an okay product, nothing special.",
        "I'm disappointed with the quality.",
        "Great value for money, highly recommended!",
        "The customer service was excellent.",
        "Not sure how I feel about this product.",
        "It broke after a week, very poor quality.",
        "Exactly what I was looking for!",
        "Meh, it's alright I guess."
    ]

    sample_sequences = tokenizer.texts_to_sequences(sample_reviews)
    sample_padded = pad_sequences(sample_sequences, maxlen=200, padding='post', truncating='post')
    sample_predictions = model.predict(sample_padded)
    sample_pred_classes = np.argmax(sample_predictions, axis=1)

    sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
    print("\nTesting model on sample reviews:")
    for review, pred_class in zip(sample_reviews, sample_pred_classes):
        print(f"Review: {review}")
        print(f"Predicted sentiment: {sentiment_map[pred_class]}\n")