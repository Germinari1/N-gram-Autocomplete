# N-gram Language Model with Kneser-Ney Smoothing

This project implements an N-gram language model with Kneser-Ney smoothing. It's designed to predict the next word in a sequence based on the previous N-1 words, and in the process demonstrate with a simple implementation how probabilistic models can be used for language-related applications based on mathematical principles that are simple to understand.

## Features

- Implements N-gram language model with configurable N
- Uses Kneser-Ney smoothing for better handling of unseen n-grams
- Includes data preprocessing and model training utilities
- Supports model saving and loading
- Provides a simple interface for getting word suggestions and generating sentences

## Files

- `main.py`: The entry point of the application. Demonstrates model usage and allows user interaction.
- `ngram_model.py`: Contains the core `NgramModel` class implementation.
- `preprocessor.py`: Handles data preprocessing, including tokenization and vocabulary building.
- `trainUtil.py`: Provides utility functions for training, data loading, and model persistence.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Germinari1/N-gram-Autocomplete.git 
   cd ngram-language-model
   ```

2. Install the required dependencies:
   ```
   pip install nltk
   ```

3. Download the necessary NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

## Usage
Run the main script:
   ```
   python main.py
   ```
   This will:
   - Train a new model if no saved model is found
   - Demonstrate sentence completion with example phrases
   - Allow you to input your own phrases for prediction

## Customization

You can modify the following parameters in `main.py`:
- `train_percentage`: The proportion of data to use for training (default: 0.8)
- `min_freq`: The minimum frequency for a word to be included in the vocabulary (default: 2)
- `n`: The maximum n-gram size (default: 3)
