# src/data_preprocessing.py

import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, '..', file_path)
    print(f"Attempting to load file from: {full_path}")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file {full_path} does not exist.")
    data = pd.read_csv(full_path)
    return data

def preprocess_data(data, max_words=10000, max_len=200):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['review'])
    sequences = tokenizer.texts_to_sequences(data['review'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer