#################################################################
# Utility functions for training, preprocessing data, and loading the ngram model
# Notes:
#################################################################
import random
from preprocessor import Preprocessor
from ngram_model import NgramModel
import pickle

######## CONSTANTS ######## 
MODEL_FILE = "ngram_model.pkl"

######## UTIL FUNCTIONS ########
def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def get_trained_model(corpusFile, train_percentage=0.8, min_freq=2, n=3):
    raw_data = load_data(corpusFile)
    preprocessor = Preprocessor()
    tokenized_data = preprocessor.get_tokenized_data(raw_data)
    
    #split data
    random.shuffle(tokenized_data)
    train_size = int((len(tokenized_data)) * train_percentage)
    train_data = tokenized_data[:train_size]
    test_data = tokenized_data[train_size:]
    
    #process data
    train_data_processed, test_data_processed, vocabulary = preprocessor.preprocess_data(train_data, test_data, min_freq)
    
    #get model
    ngram_model = NgramModel(train_data_processed, vocabulary, max_n=n)
    
    return ngram_model

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)