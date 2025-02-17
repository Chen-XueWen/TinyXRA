import argparse
import pandas as pd
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from tqdm.auto import tqdm

def sent_word_counter(data_dir: str, test_year: int = 2024) -> None:
    """
    Reads a labelled json, tokenizes the text into sentences
    and words, counts them, and prints the 80th percentile of the sentence and word counts.

    Args:
        data_dir (str): The directory where the json file is located.
        test_year (int): Test Year.
    """
    # Lists to store counts for each row
    sentence_counts = []
    word_counts = []
    train_text = []

    train_years = range(test_year-1, test_year-6, -1)
    for year in train_years:
        df = pd.read_json(f"{data_dir}/{year}.json")
        mdna_list = df['MD&A'].tolist()
        train_text += mdna_list
    
    for text in tqdm(train_text, total=len(train_text)):
        # Tokenize the concatenated text into sentences and words.
        sentences = sent_tokenize(text)
        for sent in sentences:
            words = word_tokenize(sent)
            num_words = len(words)
            word_counts.append(num_words)
        num_sentences = len(sentences)
        # Store the counts for later percentile calculation.
        sentence_counts.append(num_sentences)
        
    # Calculate the 80th percentile for sentence counts and word counts.
    sentences_80th = np.percentile(sentence_counts, 80)
    words_80th = np.percentile(word_counts, 80)

    print(f"80th Percentile Sentence Count: {sentences_80th}")
    print(f"80th Percentile Word Count: {words_80th}")


def main():
    parser = argparse.ArgumentParser(
        description="Process a test year and training years json."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/labelled",
    )
    parser.add_argument(
        "--test_year",
        type=int,
        default="2024",
    )

    args = parser.parse_args()
    sent_word_counter(args.data_dir, args.test_year)

if __name__ == "__main__":
    main()