import argparse
import csv
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from tqdm.auto import tqdm

def sent_word_counter(data_dir: str, filename: str = "train.csv") -> None:
    """
    Reads a CSV file, concatenates the first two columns of each row, tokenizes the text into sentences
    and words, counts them, and prints the 80th percentile of the sentence and word counts.

    Args:
        data_dir (str): The directory where the CSV file is located.
        filename (str): The name of the CSV file. Defaults to "train.csv".
    """
    # Lists to store counts for each row
    sentence_counts = []
    word_counts = []
    labels = []

    file_path = os.path.join(data_dir, filename)
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader):
            # Concatenate the title and text columns.
            if len(row) == 3:
                combined_text = f"{row[1]} {row[2]}"
            elif len(row) == 2:
                combined_text = f"{row[1]}"

            # Tokenize the concatenated text into sentences and words.
            sentences = sent_tokenize(combined_text)
            words = word_tokenize(combined_text)

            # Count the number of sentences and words.
            num_sentences = len(sentences)
            num_words = len(words)

            # Store the counts for later percentile calculation.
            sentence_counts.append(num_sentences)
            word_counts.append(num_words)
            labels.append(int(row[0]))

    # Calculate the 80th percentile for sentence counts and word counts.
    sentences_80th = np.percentile(sentence_counts, 80)
    words_80th = np.percentile(word_counts, 80)
    unique_labels = set(labels)

    print(f"80th Percentile Sentence Count: {sentences_80th}")
    print(f"80th Percentile Word Count: {words_80th}")
    print(f"Labels: {unique_labels}")
    print(f"# Unique Labels:{len(unique_labels)}")

def main():
    parser = argparse.ArgumentParser(
        description="Process a CSV file to compute sentence and word counts for the concatenated first two columns."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/",
        help="Directory of the CSV file (e.g., '../data/')."
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="ag_news",
        help="Directory of the CSV file (e.g., 'ag_news, dbpedia', yelp_review_full)."
    )

    args = parser.parse_args()
    data_loc = args.data_dir + args.data_type + "_csv"
    sent_word_counter(data_loc)

if __name__ == "__main__":
    main()