"""
Sentiment Analysis Using CNN:

This script loads a pre-trained CNN model for sentiment analysis and uses it to
analyze tweets retrieved from Twitter.

Key Components:
- Loads a sentiment analysis CNN model.
- Fetches tweets based on a user query.
- Processes tweets and predicts sentiment scores.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from sentiment_analysis import preprocess_tweet, text_to_indices, SentimentCNN, w2v_model, CNN_MODEL_PATH, fetch_tweets

"""
Model Configuration:
- Defines hyperparameters for dropout rate, filter sizes, number of filters, embedding dimensions, and vocabulary size.
- MAX_LENGTH sets the maximum token length for input tweets.
- MAX_RESULTS specifies the number of tweets to analyze.
"""
DROPOUT = 0.3
NUM_FILTERS = [128, 128, 128]
FILTER_SIZES = [3, 5, 7]  #
NUM_CLASSES = 1
MAX_LENGTH = 34

VOCAB_SIZE = len(w2v_model.key_to_index)
EMBEDDING_DIM = 300
MAX_RESULTS = 10

"""
Device Selection:
- Checks for GPU availability and sets computation device accordingly.
- Uses CUDA if available; otherwise, defaults to CPU.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_model(model_path, device):
    """
    Loads a pre-trained SentimentCNN model.

    Args:
    - model_path (str): Path to the saved model file.
    - device (torch.device): The computation device (CPU or GPU).

    Returns:
    - model (SentimentCNN): Loaded sentiment analysis model.
    """
    print("Creating SentimentCNN model...")
    model = SentimentCNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        dropout=DROPOUT,
        embedding_model=w2v_model
    ).to(device)
    print("Model created successfully.")

    # Load model parameters if the file exists
    if os.path.exists(model_path):
        print("Loading model parameters...")
        model.load_state_dict(torch.load(model_path, weights_only=True)['model_state_dict'])
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        return model
    else:
        print(f"Model parameters file '{model_path}' not found.")
        return None


def analyze_tweets(model, tweets, device):
    """
    Analyzes sentiment for a list of tweets using the trained model.

    Args:
    - model (SentimentCNN): The loaded sentiment analysis model.
    - tweets (list): List of tweet texts.
    - device (torch.device): The computation device (CPU or GPU).

    Returns:
    - sentiments (list): List of sentiment scores for each tweet.
    - average_sentiment (float): Average sentiment score of all tweets.
    """
    sentiments = []

    for idx, tweet in enumerate(tweets, 1):
        # Preprocess and tokenize tweet
        tokenized_tweet = [preprocess_tweet(tweet)]
        input_tensor = torch.tensor(text_to_indices(tokenized_tweet, MAX_LENGTH), dtype=torch.long).to(device)

        print(f"Tokenized tweet: {tokenized_tweet}")
        print(f"Indexed tweet: {input_tensor}")

        with torch.no_grad():
            # Get sentiment score from model
            sentiment_logits = model(input_tensor)
            print(f"Raw logits for Tweet {idx}: {sentiment_logits.cpu().numpy()}")

            # Convert logits to probability using sigmoid activation
            probability = F.sigmoid(sentiment_logits).cpu().numpy()[0]
            print(f"Sigmoid probability for Tweet {idx}: {probability}")

            # Categorize sentiment based on probability range
            if 0 <= probability < 0.2:
                sentiment_category = "Very Negative"
            elif 0.2 <= probability < 0.4:
                sentiment_category = "Negative"
            elif 0.4 <= probability < 0.6:
                sentiment_category = "Neutral"
            elif 0.6 <= probability < 0.8:
                sentiment_category = "Positive"
            elif 0.8 <= probability <= 1:
                sentiment_category = "Very Positive"
            else:
                sentiment_category = "Unknown"

            sentiments.append(probability)

        print(f"Tweet {idx}: {tweet}")
        print(f"Predicted positivity score: {probability}")
        print(f"Sentiment: {sentiment_category}\n")

    """
    Compute and display average sentiment score:
    - If there are analyzed tweets, compute the mean score.
    - If no tweets were analyzed, default to 0.
    """

    if sentiments:
        average_sentiment = np.mean(sentiments).item()
        print(f"Average Sentiment Score: {float(average_sentiment):.2f}")
    else:
        average_sentiment = 0
        print("No tweets to analyze.")

    return sentiments, average_sentiment


if __name__ == "__main__":
    """
    Main Execution:
    - Loads the trained sentiment analysis model.
    - Prompts the user for a search query.
    - Fetches tweets based on the query.
    - Analyzes the sentiment of fetched tweets.
    """

    model = load_model(CNN_MODEL_PATH, device)

    query = input("Enter a query string to search tweets: ")

    print(f"Fetching tweets for query: '{query}'...")
    tweets = fetch_tweets(query, MAX_RESULTS)

    analyze_tweets(model, tweets, device)
