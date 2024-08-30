#################################################################
# A class to preprocess data for a n-gram language model
# Notes:
#################################################################

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def split_to_sentences(self, data):
        sentences = data.split('\n')
        
        # Additional clearning (This part is already implemented)
        # - Remove leading and trailing spaces from each sentence
        # - Drop sentences if they are empty strings.
        sentences = [s.strip() for s in sentences]
        sentences = [s for s in sentences if len(s) > 0]
        
        return sentences

    def tokenize_sentences(self, sentences):
        tokenized_sentences = []
        
        # Go through each sentence
        for sentence in sentences:
            
            # Convert to lowercase letters
            sentence = sentence.lower()
            
            # Convert into a list of words
            tokenized = nltk.word_tokenize(sentence)
            
            # append the list of words to the list of lists
            tokenized_sentences.append(tokenized)
        
        return tokenized_sentences


    def get_tokenized_data(self, data):
        sentences = self.split_to_sentences(data)
        tokenized_sentences = self.tokenize_sentences(sentences)
        return tokenized_sentences

    def count_words(self, tokenized_sentences):
        word_counts = {}
        for sentence in tokenized_sentences:
            for token in sentence:
                if token not in word_counts:
                    word_counts[token] = 1
                else:
                    word_counts[token] += 1
        return word_counts

    def get_words_with_nplus_frequency(self, tokenized_sentences, count_threshold):
        closed_vocab = []
        word_counts = self.count_words(tokenized_sentences)
        for word, cnt in word_counts.items():
            if cnt >= count_threshold:
                closed_vocab.append(word)
        return closed_vocab

    def replace_oov_words_by_unk(self, tokenized_sentences, vocabulary, unknown_token="<unk>"):
        vocabulary = set(vocabulary)
        replaced_tokenized_sentences = []
        for sentence in tokenized_sentences:
            replaced_sentence = []
            for token in sentence:
                if token in vocabulary:
                    replaced_sentence.append(token)
                else:
                    replaced_sentence.append(unknown_token)
            replaced_tokenized_sentences.append(replaced_sentence)
        return replaced_tokenized_sentences

    def preprocess_data(self, train_data, test_data, count_threshold):
        vocabulary = self.get_words_with_nplus_frequency(train_data, count_threshold)
        train_data_replaced = self.replace_oov_words_by_unk(train_data, vocabulary)
        test_data_replaced = self.replace_oov_words_by_unk(test_data, vocabulary)
        return train_data_replaced, test_data_replaced, vocabulary