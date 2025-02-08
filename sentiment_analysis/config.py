from gensim.models import KeyedVectors

VERSION_NUMBER = 1

W2V_MODEL_PATH = '../datasets/GoogleNews-vectors-negative300.bin'
SENTIMENT140_FILE_PATH = f'../datasets/Sentiment140v{VERSION_NUMBER}.csv'
CNN_MODEL_PATH = '../models/CNN_model.pth'

try:
    w2v_model = KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=True)
except FileNotFoundError:
    print("Error: Model file not found at the specified path.")
except Exception as e:
    print(f"An error occurred: {e}")
