# src/model.py

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def build_model(vocab_size, embedding_dim=50, max_len=200):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Build the model with some sample data
    sample_data = np.random.randint(0, vocab_size, size=(1, max_len))
    model(sample_data)
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, class_weight_dict, epochs=20, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=1,
                        class_weight=class_weight_dict,
                        callbacks=[early_stopping])
    return history

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return loss, accuracy