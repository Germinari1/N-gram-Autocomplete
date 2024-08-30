#################################################################
# Uses somes example sentences to show the model's predictions, and allows the user to input their own sentences to get predictions.
# Note:
    # If a saved model exists, it is loaded. Otherwise, a new model is trained.
    # The model is trained on the "en_US.twitter.txt" dataset, and its predictions will reflect this dataset linguistic caracteristics.
#################################################################

import os
import pickle
from preprocessor import Preprocessor
from ngram_model import NgramModel
from trainUtil import *

def main():
    # Check if a saved model exists
    if os.path.exists(MODEL_FILE):
        print("Loading saved model...")
        ngram_model = load_model(MODEL_FILE)
    else:
        print("No saved model found. Training new model...")
        ngram_model = get_trained_model("en_US.twitter.txt", train_percentage=0.8, min_freq=2, n=3)
        save_model(ngram_model, MODEL_FILE)
    
    # Show examples of sentence completion
    print("\nExamples of sentence completion:")
    example_phrases = [
        ["i", "am"],
        ["how", "are", "you"],
        ["what", "is", "the"],
        ["have", "a", "nice"]
    ]
    for phrase in example_phrases:
        suggestions = ngram_model.get_suggestions(phrase, k=3)
        print(f"Phrase: {' '.join(phrase)}")
        for word, prob in suggestions:
            print(f"  '{word}' with probability {prob:.4f}")
        print()

    # User input for sentence completion
    while True:
        user_input = input("Enter a short sentence (or 'q' to quit): ").lower().split()
        if user_input[0] == 'q':
            break
        if len(user_input) > 5:
            print("Please enter a shorter sentence (5 words or less).")
            continue
        
        suggestions = ngram_model.get_suggestions(user_input, k=3)
        print(f"Next word predictions for '{' '.join(user_input)}':")
        for word, prob in suggestions:
            print(f"  '{word}' with probability {prob:.4f}")
        print()

if __name__ == "__main__":
    main()