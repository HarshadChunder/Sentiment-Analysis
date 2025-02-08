from gensim.models import KeyedVectors
import os

VERSION_NUMBER = 1

W2V_MODEL_PATH = '../datasets/GoogleNews-vectors-negative300.bin'
SENTIMENT140_FILE_PATH = f'../datasets/Sentiment140v{VERSION_NUMBER}.csv'
CNN_MODEL_PATH = '../models/CNN_model.pth'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")

required_files = [
    os.path.join(DATA_DIR, "Sentiment140v1.csv"),
    os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")
]

def check_files():
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("The following files are missing:")
        for file in missing_files:
            print(f"- {file}")
        return False
    return True

if not check_files():
        os.system("python ../setup/setup_datasets.py")
        print("Setup files downloaded. Please re-run the script.")
        exit(1)

try:
    w2v_model = KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=True)
except FileNotFoundError:
    print("Error: Model file not found at the specified path.")
except Exception as e:
    print(f"An error occurred: {e}")
