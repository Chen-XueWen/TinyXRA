import argparse
import csv
import os
import json
import h5py
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

def save_to_hdf5(file_path, docs, labels):
    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("docs", data=np.array(docs, dtype=np.int32), compression="gzip")
        hf.create_dataset("labels", data=np.array(labels, dtype=np.int32))
    print(f"Data saved to {file_path}.")

# ==============================
# Load GloVe Embeddings & Create Vocabulary
# ==============================
def load_glove_embeddings(glove_path="./glove.6B.300d.txt", embedding_dim=300):
    """
    Loads GloVe embeddings and creates word-to-index mapping.

    Args:
        glove_path (str): Path to GloVe embedding file.
        embedding_dim (int): Embedding dimension.

    Returns:
        word2idx (dict): Dictionary mapping words to unique indices.
        idx2word (dict): Reverse mapping (indices to words).
        embeddings_matrix (np.array): Matrix of preloaded embeddings.
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}  # Special tokens
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    embeddings = [
        np.zeros(embedding_dim, dtype=np.float32),  # PAD token vector
        np.random.uniform(-0.1, 0.1, embedding_dim).astype(np.float32),  # UNK token vector
    ]  # Pad & OOV

    with open(glove_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=2):
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            word2idx[word] = i
            idx2word[i] = word
            embeddings.append(vector)

    return word2idx, idx2word, np.array(embeddings, dtype=np.float32)

def preprocess_text(text, max_sentences, max_words, word2idx):
    """
    Tokenizes text into sentences and words, then replaces words with their indices.

    Args:
        text (str): Input text.
        max_sentences (int): Max sentences per document.
        max_words (int): Max words per sentence.
        word2idx (dict): Mapping from words to indices.

    Returns:
        np.array: Padded document representation (shape: max_sentences x max_words).
    """
    sentences = sent_tokenize(text.lower())  # Tokenize to sentences
    processed_sentences = []

    for sent in sentences[:max_sentences]:  # Limit sentences
        words = word_tokenize(sent)[:max_words]  # Tokenize & limit words
        word_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in words]  # Convert words to indices

        # Pad sentence if needed
        while len(word_indices) < max_words:
            word_indices.append(word2idx["<PAD>"])

        processed_sentences.append(word_indices)

    # Pad document if needed
    while len(processed_sentences) < max_sentences:
        processed_sentences.append([word2idx["<PAD>"]] * max_words)

    return np.array(processed_sentences, dtype=np.int32)

def process_data(data_dir, test_year, max_sentences, max_words, word2idx, training):
    processed_docs = []
    labels = []
    
    if training == True:
        years = range(test_year-1, test_year-6, -1)
    else:
        years = [test_year]
    
    for year in years:
        print(f"Processing Year {year}")
        df = pd.read_json(f"{data_dir}/{year}.json")
        mdna_list = df['MD&A'].tolist()
        cik_list = df['CIK']
        Std_labels = df["Std_label"].tolist()
        Skewness_labels = df["Skewness_label"].tolist()
        Kurtosis_labels = df["Kurtosis_label"].tolist()
        Sortino_labels = df["Sortino_label"].tolist()

        for (cik, mdna, std, skew, kurt, sortino) in tqdm(zip(cik_list, mdna_list, Std_labels, Skewness_labels, Kurtosis_labels, Sortino_labels), total=len(mdna_list)):
            processed = preprocess_text(mdna, max_sentences, max_words, word2idx)
            processed_docs.append(processed)
            labels.append((int(std), int(skew), int(kurt), int(sortino)))

    return processed_docs, labels


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess train and test data for Hierarchical Attention Network."
    )
    parser.add_argument(
        "--test_year",
        type=int,
        default="2024",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../datasets/labelled",
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=200,
        help="Maximum number of sentences per document (e.g., ~ Half the 80th percentile sentence count, assuming first half are more important)."
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=30,
        help="Maximum number of words per sentence (e.g., ~ 80th percentile word count)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="glove6B300d",
        help="Model name for AutoTokenizer (default: albert-base-v2)."
    )
    
    args = parser.parse_args()
    output_dir = f"../processed/{args.test_year}/{args.model_name}"
    # Check if output_dir exists, else create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    # Check if word2idx and embeddings exist
    if os.path.exists("./word2idx.json") and os.path.exists("./glove_embeddings.npy"):
        print("Loading existing word2idx and GloVe embeddings...")
        with open("./word2idx.json", "r") as f:
            word2idx = json.load(f)
        print("Existing embeddings loaded.")
    else:
        # Load GloVe and get vocabulary
        print("Creating word2idx.json and glove_embeddings.npy")
        print("Loading GloVe embeddings...")
        word2idx, idx2word, embeddings_matrix = load_glove_embeddings('./glove.6B.300d.txt', 300)
        print("GloVe embeddings loaded.")
        with open("./word2idx.json", "w") as f:
            json.dump(word2idx, f)
        np.save("./glove_embeddings.npy", embeddings_matrix)

    print("Processing training data...")
    train_docs, train_labels = process_data(data_dir=args.data_dir, 
                                            test_year=args.test_year, 
                                            max_sentences=args.max_sentences, 
                                            max_words=args.max_words, 
                                            word2idx=word2idx, 
                                            training=True)
    print("Processing testing data...")
    test_docs, test_labels = process_data(data_dir=args.data_dir, 
                                          test_year=args.test_year, 
                                          max_sentences=args.max_sentences, 
                                          max_words=args.max_words, 
                                          word2idx=word2idx, 
                                          training=False)
    
    save_to_hdf5(os.path.join(output_dir, "train_preprocessed.h5"), train_docs, train_labels)
    save_to_hdf5(os.path.join(output_dir, "test_preprocessed.h5"), test_docs, test_labels)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()