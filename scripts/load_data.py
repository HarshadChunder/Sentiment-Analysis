import pandas as pd
from sentiment_analysis import SENTIMENT140_FILE_PATH

VERSION_NUMBER = 2
SAMPLE_SIZE = 400000

def load_sentiment140_data():
    """Loads Sentiment140 dataset, extracting only the target and text columns."""
    data = pd.read_csv(SENTIMENT140_FILE_PATH, encoding='ISO-8859-1', header=None, names=['target', 'id', 'date', 'flag', 'user', 'text'])
    data = data[['target', 'text']]
    return data['target'], data['text']

def reduce_dataset():
    """Reduces the dataset to SAMPLE_SIZE positive and negative samples, then shuffles and saves."""
    data = pd.read_csv(SENTIMENT140_FILE_PATH, encoding='ISO-8859-1', header=None,
                       names=['target', 'id', 'date', 'flag', 'user', 'text'])

    negative = data[data['target'] == 0]
    positive = data[data['target'] == 4]

    negative_sample = negative.sample(n=SAMPLE_SIZE, random_state=42)
    positive_sample = positive.sample(n=SAMPLE_SIZE, random_state=42)

    reduced_data = pd.concat([negative_sample, positive_sample]).sample(frac=1, random_state=42)

    reduced_data.to_csv(f'../datasets/Sentiment140v{VERSION_NUMBER}.csv', index=False)

    print("Dataset saved to ../datasets/Sentiment140v2.csv")

def check_dataset():
    """Checks the number of positive and negative sentiment records in the dataset."""
    df = pd.read_csv(SENTIMENT140_FILE_PATH)

    negative_count = df[df['target'] == 0].shape[0]
    positive_count = df[df['target'] == 4].shape[0]

    print(f"Number of records with sentiment 0 (very negative): {negative_count}")
    print(f"Number of records with sentiment 4 (very positive): {positive_count}")
