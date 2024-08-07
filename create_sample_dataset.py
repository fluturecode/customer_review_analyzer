import pandas as pd
import numpy as np

# Function to generate random reviews
def generate_review(sentiment):
    positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic']
    negative_words = ['terrible', 'awful', 'disappointing', 'poor', 'bad']
    neutral_words = ['okay', 'average', 'mediocre', 'acceptable', 'fair']
    
    if sentiment == 1:
        words = np.random.choice(positive_words, size=np.random.randint(3, 6))
    elif sentiment == 0:
        words = np.random.choice(negative_words, size=np.random.randint(3, 6))
    else:
        words = np.random.choice(neutral_words, size=np.random.randint(3, 6))
    
    return ' '.join(words)

# Generate sample data
num_samples = 1000
sentiments = np.random.choice([0, 1, 2], size=num_samples)
reviews = [generate_review(s) for s in sentiments]

# Create DataFrame
df = pd.DataFrame({
    'review': reviews,
    'sentiment': sentiments
})

# Save to CSV
df.to_csv('data/raw_data.csv', index=False)
print("Sample dataset created and saved as 'data/raw_data.csv'")

# Display first few rows
print(df.head())