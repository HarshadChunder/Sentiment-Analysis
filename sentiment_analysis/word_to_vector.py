import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from .config import w2v_model

# Download NLTK resources (if needed)
#import nltk
#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('stopwords')

# Constants
STOP_WORDS = set(stopwords.words('english'))
EMBEDDING_DIM = w2v_model.vector_size

# Ensure <unk> (unknown token) exists in the word2vec model for handling unseen words
if '<unk>' not in w2v_model.key_to_index:
    w2v_model.key_to_index['<unk>'] = len(w2v_model.key_to_index)
    w2v_model.index_to_key.append('<unk>')
    w2v_model.vectors = np.vstack([w2v_model.vectors, np.random.randn(EMBEDDING_DIM)])

def preprocess_tweet(tweet):
    """
    Cleans and preprocesses a given tweet by removing unnecessary characters,
    converting it to lowercase, tokenizing it, and removing stopwords.

    Args:
        tweet (str): The tweet text to preprocess.

    Returns:
        str: The cleaned and tokenized tweet as a single string.
    """
    # Remove retweet indicators
    tweet = re.sub(r"RT\s@\w+:", "", tweet)

    # Remove retweet indicators
    tweet = re.sub(r"http\S+", "", tweet)

    # Remove special characters, numbers, and punctuation
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)

    #Convert text to lowercase
    tweet = tweet.lower()

    # Tokenize text into words
    tokens = word_tokenize(tweet)

    # Remove stopwords
    tokens = [word for word in tokens if word not in STOP_WORDS]

    # Join the cleaned tokens back into a sentence
    sentence = ' '.join(tokens)

    return sentence

def text_to_indices(texts, max_sequence_length):
    """
       Converts a list of texts into lists of word indices based on a pretrained Word2Vec model.

       Args:
           texts (list of str): The input text data.
           max_sequence_length (int): The fixed sequence length for padding/truncation.

       Returns:
           list of list of int: A list of lists where each inner list contains word indices.
       """
    indices = []

    for text in texts:
        tokens = text.split()
        token_indices = []

        for token in tokens:
            if token in w2v_model:
                token_indices.append(w2v_model.key_to_index[token])
            else:
                token_indices.append(w2v_model.key_to_index['<unk>'])

        token_indices = token_indices[:max_sequence_length]
        token_indices = token_indices + [0] * (max_sequence_length - len(token_indices))
        indices.append(token_indices)

    return indices

