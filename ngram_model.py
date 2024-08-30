#################################################################
# Implements an N-gram language model with Kneser-Ney smoothing
# Notes:
#################################################################
import math
from collections import defaultdict

class NgramModel:
    def __init__(self, train_data, vocabulary, max_n=5):
        self.train_data = train_data
        self.vocabulary = vocabulary
        self.max_n = max_n
        self.n_gram_counts_list = self._build_n_gram_counts()
        self.continuation_counts = self._build_continuation_counts()

    def _build_n_gram_counts(self):
        n_gram_counts_list = []
        for n in range(1, self.max_n + 1):
            print(f"Computing {n}-gram counts...")
            n_model_counts = self._count_n_grams(self.train_data, n)
            n_gram_counts_list.append(n_model_counts)
        return n_gram_counts_list

    def _count_n_grams(self, data, n, start_token='<s>', end_token='<e>'):
        n_grams = defaultdict(int)
        for sentence in data:
            padded_sentence = [start_token] * (n-1) + sentence + [end_token]
            for i in range(len(padded_sentence) - n + 1):
                n_gram = tuple(padded_sentence[i:i+n])
                n_grams[n_gram] += 1
        return n_grams

    def _build_continuation_counts(self):
        continuation_counts = []
        for n in range(1, self.max_n):
            print(f"Computing continuation counts for {n}-grams...")
            count_dict = defaultdict(int)
            n_plus1_grams = self.n_gram_counts_list[n]
            for n_plus1_gram in n_plus1_grams:
                n_gram = n_plus1_gram[:-1]
                count_dict[n_gram] += 1
            continuation_counts.append(count_dict)
        return continuation_counts

    def _kneser_ney_smoothing(self, word, previous_n_gram, n, d=0.75):
        if n == 1:  # Unigram case
            return self._estimate_unigram_probability(word)
        
        n_gram_counts = self.n_gram_counts_list[n-1]
        n_minus1_gram_counts = self.n_gram_counts_list[n-2]
        continuation_counts = self.continuation_counts[n-2]

        previous_n_gram = tuple(previous_n_gram)
        n_gram = previous_n_gram + (word,)

        count_n_gram = n_gram_counts.get(n_gram, 0)
        count_prev = n_minus1_gram_counts.get(previous_n_gram, 0)
        
        if count_prev == 0:
            return self._kneser_ney_smoothing(word, previous_n_gram[1:], n-1, d)

        continuation_count = continuation_counts[previous_n_gram]
        
        lambda_factor = d * continuation_count / count_prev
        higher_order_prob = max(count_n_gram - d, 0) / count_prev
        lower_order_prob = self._kneser_ney_smoothing(word, previous_n_gram[1:], n-1, d)

        return higher_order_prob + lambda_factor * lower_order_prob

    def _estimate_unigram_probability(self, word):
        unigram_counts = self.n_gram_counts_list[0]
        total_words = sum(unigram_counts.values())
        return unigram_counts.get((word,), 0) / total_words

    def calculate_perplexity(self, sentence, n):
        if n > self.max_n:
            raise ValueError(f"n should be less than or equal to {self.max_n}")
        
        sentence = ["<s>"] * (n-1) + sentence + ["<e>"]
        N = len(sentence)
        log_probability_sum = 0.0
        epsilon = 1e-10  # Small value to avoid log(0)

        for t in range(n-1, N):
            previous_n_gram = tuple(sentence[t-(n-1):t])
            word = sentence[t]
            probability = self._kneser_ney_smoothing(word, previous_n_gram, n)
            log_probability_sum += math.log(max(probability, epsilon))

        perplexity = math.exp(-log_probability_sum / float(N-n+1))
        return perplexity

    def get_suggestions(self, previous_tokens, k=5, start_with=None):
        n = min(len(previous_tokens) + 1, self.max_n)
        previous_n_gram = tuple(previous_tokens[-(n-1):])
        
        probabilities = {}
        for word in self.vocabulary:
            if start_with and not word.startswith(start_with):
                continue
            prob = self._kneser_ney_smoothing(word, previous_n_gram, n)
            probabilities[word] = prob

        sorted_suggestions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return sorted_suggestions[:k]

    def generate_sentence(self, max_length=20):
        sentence = ["<s>"]
        for _ in range(max_length):
            suggestions = self.get_suggestions(sentence)
            next_word, _ = suggestions[0]  # Get the most probable word
            if next_word == "<e>":
                break
            sentence.append(next_word)
        return sentence[1:]  # Remove the start token

    def save_model(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'train_data': self.train_data,
                'vocabulary': self.vocabulary,
                'max_n': self.max_n,
                'n_gram_counts_list': self.n_gram_counts_list,
                'continuation_counts': self.continuation_counts
            }, f)

    @classmethod
    def load_model(cls, filename):
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        model = cls(data['train_data'], data['vocabulary'], data['max_n'])
        model.n_gram_counts_list = data['n_gram_counts_list']
        model.continuation_counts = data['continuation_counts']
        return model