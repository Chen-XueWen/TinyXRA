import argparse
import os
import h5py
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

def save_to_hdf5(file_path, docs, attn_masks, tfidf_weights, labels):
    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("docs", data=np.array(docs, dtype=np.int32), compression="gzip")
        hf.create_dataset("attn_masks", data=np.array(attn_masks, dtype=np.int32), compression="gzip")
        hf.create_dataset("tfidf_weights", data=np.array(tfidf_weights, dtype=np.float32), compression="gzip")
        hf.create_dataset("labels", data=np.array(labels, dtype=np.int32), compression="gzip")
    print(f"Data saved to {file_path}.")


def preprocess_text(text, max_sentences, max_words, tokenizer, tfidf_vectorizer=None):
    """
    Tokenizes text into sentences using NLTK and then uses the AutoTokenizer to encode
    each sentence into token IDs. Each sentence is truncated (or padded) to have exactly
    max_words tokens, and the document is truncated (or padded) to have exactly max_sentences sentences.
    
    Args:
        text (str): The input text to be tokenized.
        max_sentences (int): Maximum number of sentences per document.
        max_words (int): Maximum number of tokens per sentence.
        tokenizer (AutoTokenizer): A pre-initialized AutoTokenizer.
    
    Returns:
        processed_sentences: List[List[int]]
        attention_mask:       List[List[int]]
        tfidf_weights:        List[List[float]]
    """
    # Tokenize the document into sentences.
    sentences = sent_tokenize(text)
    processed_sentences = []
    attention_mask = []
    tfidf_weights = []
    
    # skip first 3 sentences (MD&A title), then take up to max_sentences
    for sent in sentences[3: 3 + max_sentences]:
        enc = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=max_words,
            truncation=True,
            padding="max_length"
        )
        input_ids = enc["input_ids"]
        mask      = enc["attention_mask"]
        processed_sentences.append(input_ids)
        attention_mask.append(mask)
        
        # --- token‐based TF–IDF lookup ---
        if tfidf_vectorizer is not None:
            # `vector` is a 1×V sparse TF–IDF array
            vector = tfidf_vectorizer.transform([sent]).toarray()[0]
            # convert IDs back to tokens: ["[CLS]", "the", "men", …, "[SEP]", "[PAD]", …]
            toks   = tokenizer.convert_ids_to_tokens(input_ids)
            weights = [
                float(vector[ tfidf_vectorizer.vocabulary_.get(tok, -1) ])
                if tok in tfidf_vectorizer.vocabulary_ else 0.0
                for tok in toks
            ]
        else:
            weights = [0.0] * len(input_ids)
        tfidf_weights.append(weights)
    
    # pad up to max_sentences
    pad_ids   = [tokenizer.pad_token_id] * max_words
    pad_mask  = [0] * max_words
    pad_tfidf = [0.0] * max_words
    while len(processed_sentences) < max_sentences:
        processed_sentences.append(pad_ids)
        attention_mask.append(pad_mask)
        tfidf_weights.append(pad_tfidf)
    
    return processed_sentences, attention_mask, tfidf_weights

def process_data(data_dir, test_year, max_sentences, max_words, tokenizer, tfidf_vectorizer=None, training=True):
    docs, masks, tfidf_all, labels = [], [], [], []
    
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
            p, m, w = preprocess_text(mdna, max_sentences, max_words, tokenizer, tfidf_vectorizer)
            docs.append(p)
            masks.append(m)
            tfidf_all.append(w)
            labels.append((int(std), int(skew), int(kurt), int(sortino)))

    return docs, masks, tfidf_all, labels


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
        default=350,
        help="Maximum number of sentences per document (e.g., ~ 80th percentile sentence count)."
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=40,
        help="Maximum number of words per sentence (e.g., ~ 80th percentile word count)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="huawei-noah/TinyBERT_General_4L_312D",
        help="Model name for AutoTokenizer (default: albert-base-v2)."
    )
    args = parser.parse_args()
    output_dir = f"../processed/{args.test_year}/tfidfhans"

    # Check if output_dir exists, else create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 1) fit TF–IDF on TinyBERT tokens
    print("Fitting TF-IDF on TinyBERT-tokenized text…")
    raw_texts = []
    for yr in range(args.test_year-1, args.test_year-6, -1):
        df = pd.read_json(f"{args.data_dir}/{yr}.json")
        raw_texts.extend(df["MD&A"].tolist())

    tfidf_vectorizer = TfidfVectorizer(
        analyzer=tokenizer.tokenize,
        lowercase=False,
        token_pattern=None,
    )
    tfidf_vectorizer.fit(raw_texts)

    # 2) process train & test, now passing our vectorizer
    train_docs, train_masks, train_tfidf, train_labels = process_data(
        data_dir=args.data_dir, test_year=args.test_year,
        max_sentences=args.max_sentences, max_words=args.max_words,
        tokenizer=tokenizer, tfidf_vectorizer=tfidf_vectorizer,
        training=True
    )
    test_docs, test_masks, test_tfidf, test_labels = process_data(
        data_dir=args.data_dir, test_year=args.test_year,
        max_sentences=args.max_sentences, max_words=args.max_words,
        tokenizer=tokenizer, tfidf_vectorizer=tfidf_vectorizer,
        training=False
    )

    # 3) save including tfidf_weights
    save_to_hdf5(
        os.path.join(output_dir, "train_preprocessed.h5"),
        train_docs, train_masks, train_tfidf, train_labels
    )
    save_to_hdf5(
        os.path.join(output_dir, "test_preprocessed.h5"),
        test_docs, test_masks, test_tfidf, test_labels
    )

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()