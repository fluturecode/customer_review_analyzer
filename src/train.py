import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
import matplotlib.pyplot as plt

# Step 1: Data Collection
def load_data(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, file_path)
    print(f"Attempting to load file from: {full_path}")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file {full_path} does not exist.")
    data = pd.read_csv(full_path)
    return data

# Step 2: Data Preprocessing
def preprocess_data(data, max_words=10000, max_len=200):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['review'])
    sequences = tokenizer.texts_to_sequences(data['review'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer

# Step 3: Building the Model
def build_model(vocab_size, embedding_dim=100, max_len=200):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Build the model with some sample data
    sample_data = np.random.randint(0, vocab_size, size=(1, max_len))
    model(sample_data)
    
    return model

# Step 4: Training the Model
def train_model(model, X_train, y_train, X_test, y_test, class_weight_dict, epochs=50, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=1,
                        class_weight=class_weight_dict,
                        callbacks=[early_stopping])
    return history

# Step 5: Evaluating the Model
def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    precision = precision_score(y_test, y_pred_classes, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Main execution
if __name__ == "__main__":
    # Load data
    data = load_data('../data/raw_data.csv')
    print("Data loaded successfully. Shape:", data.shape)

    # Print sentiment distribution
    print("Sentiment distribution in training data:")
    print(data['sentiment'].value_counts(normalize=True))

    # Preprocess data
    X, tokenizer = preprocess_data(data)
    
    # Encode sentiments
    le = LabelEncoder()
    y = le.fit_transform(data['sentiment'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
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
    print("Training history:", history.history)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Plot training history
    print("Generating training history plot...")
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
    
    print("\nTesting model on sample reviews:")
sample_reviews = [
    "This product is amazing! I love it!",
    "Terrible experience, never buying again.",
    "It's an okay product, nothing special."
]

# Preprocess the sample reviews
sample_sequences = tokenizer.texts_to_sequences(sample_reviews)
sample_padded = pad_sequences(sample_sequences, maxlen=200, padding='post', truncating='post')

# Make predictions
sample_predictions = model.predict(sample_padded)
sample_pred_classes = np.argmax(sample_predictions, axis=1)

# Map predictions back to sentiment labels
sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
for review, pred_class in zip(sample_reviews, sample_pred_classes):
    print(f"Review: {review}")
    print(f"Predicted sentiment: {sentiment_map[pred_class]}\n")