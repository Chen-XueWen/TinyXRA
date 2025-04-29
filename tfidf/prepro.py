import argparse
import csv
import os
import json
import h5py
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def save_to_hdf5(file_path, docs, labels, dtype=np.float32):
    """
    Save a (num_docs x num_features) array of TF-IDF features and labels.
    """
    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("docs", data=docs.astype(dtype), compression="gzip")
        hf.create_dataset("labels", data=np.array(labels, dtype=np.int32))
    print(f"Data saved to {file_path}.")

def truncate_text(text, max_sents=350, max_words=40):
    """
    Split `text` into sentences, keep only the first `max_sents`,
    then within each sentence keep only the first `max_words` tokens.
    Return the truncated text as a single string., To make it fair when benchmark against TinyXRA and XRR
    """
    # split into sentences
    sents = sent_tokenize(text)
    truncated = []
    for sent in sents[:max_sents]:
        # split into words and truncate
        words = word_tokenize(sent)
        truncated.append(" ".join(words[:max_words]))
    # rejoin back into a document
    return " ".join(truncated)

def load_texts_and_labels(data_dir, years):
    texts, labels = [], []
    for year in years:
        print(f"  ▸ loading {year}.json")
        df = pd.read_json(os.path.join(data_dir, f"{year}.json"))
        for _, row in df.iterrows():
            # truncate the MD&A section to 350 sentences × 40 words
            truncated = truncate_text(row["MD&A"],
                                      max_sents=350,
                                      max_words=40)
            texts.append(truncated)
            labels.append((int(row["Std_label"]),
                           int(row["Skewness_label"]),
                           int(row["Kurtosis_label"]),
                           int(row["Sortino_label"])))
    return texts, labels


def build_or_load_vectorizer(train_texts, max_features, ngram_min, ngram_max, save_path=None):
    """
    Fit a TfidfVectorizer on train_texts, or load one if save_path exists.
    """
    if save_path and os.path.exists(save_path):
        print("Loading existing TF-IDF vectorizer…")
        with open(save_path, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        print("Fitting TF-IDF vectorizer…")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(ngram_min, ngram_max),
            stop_words="english"
        )
        vectorizer.fit(train_texts)
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(vectorizer, f)
    return vectorizer

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess train and test data into TF-IDF vectors."
    )
    parser.add_argument(
        "--test_year",
        type=int,
        default=2024,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../datasets/labelled",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=50000,
        help="Maximum number of TF-IDF features, BERT has 30522 in their embedding layers"
    )
    parser.add_argument(
        "--ngram_min",
        type=int,
        default=1,
        help="Min n-gram length"
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=2,
        help="Max n-gram length"
    )


    args = parser.parse_args()
    output_dir = f"../processed/{args.test_year}/tfidf_{args.max_features}f"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    print(">> Loading training raw texts…")
    train_years = list(range(args.test_year-5, args.test_year))
    train_texts, train_labels = load_texts_and_labels(args.data_dir, train_years)

    print(">> Loading testing raw texts…")
    test_texts, test_labels = load_texts_and_labels(args.data_dir, [args.test_year])

    # Fit or load TF-IDF vectorizer
    vec_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    vectorizer = build_or_load_vectorizer(
        train_texts,
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        save_path=vec_path
    )

    # Transform texts → feature matrices
    print(">> Transforming train texts to TF-IDF")
    X_train = vectorizer.transform(train_texts).toarray()
    print(">> Transforming test texts to TF-IDF")
    X_test  = vectorizer.transform(test_texts).toarray()

    save_to_hdf5(
        os.path.join(output_dir, "train_tfidf.h5"),
        X_train,
        train_labels,
        dtype=np.float32
    )
    save_to_hdf5(
        os.path.join(output_dir, "test_tfidf.h5"),
        X_test,
        test_labels,
        dtype=np.float32
    )

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()