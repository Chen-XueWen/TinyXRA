import argparse
import csv
import os
import h5py
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer

def save_to_hdf5(file_path, docs, attn_masks, labels):
    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("docs", data=np.array(docs, dtype=np.int32), compression="gzip")
        hf.create_dataset("attn_masks", data=np.array(attn_masks, dtype=np.int32))
        hf.create_dataset("labels", data=np.array(labels, dtype=np.int32))
    print(f"Data saved to {file_path}.")

def preprocess_text(text, max_sentences, max_words, tokenizer):
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
        List[List[int]]: A list of sentences (each a list of token IDs).
    """
    # Tokenize the document into sentences.
    sentences = sent_tokenize(text)
    processed_sentences = []
    for i, sent in enumerate(sentences[:max_sentences+3]): #First 3 sentences are usually the MD&A title
        if i < 3:
            continue
        else:
            # Use the AutoTokenizer to encode the sentence.
            encoding = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=max_words,
                truncation=True,
                padding='max_length'
            )
            token_ids = encoding['input_ids']
            processed_sentences.append(token_ids)
    # Pad the document if there are fewer than max_sentences.
    if len(processed_sentences) < max_sentences:
        pad_sentence = [tokenizer.pad_token_id] * max_words
        processed_sentences += [pad_sentence] * (max_sentences - len(processed_sentences))
    
    attention_mask = (np.array(processed_sentences) != tokenizer.pad_token_id).astype(int).tolist()
    return processed_sentences, attention_mask

def process_data(data_dir, test_year, max_sentences, max_words, tokenizer, training):
    processed_docs = []
    attention_masks = []
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
            processed, attention_mask = preprocess_text(mdna, max_sentences, max_words, tokenizer)
            processed_docs.append(processed)
            attention_masks.append(attention_mask)
            labels.append((int(std), int(skew), int(kurt), int(sortino)))

    return processed_docs, attention_masks, labels


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess train and test CSV data for Hierarchical Attention Network."
    )
    parser.add_argument(
        "--test_year",
        type=int,
        default="2024",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/labelled",
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
        default="albert-base-v2",
        help="Model name for AutoTokenizer (default: albert-base-v2)."
    )
    args = parser.parse_args()
    output_dir = f"./processed/{args.test_year}/{args.model_name}"

    # Check if output_dir exists, else create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Processing training data...")
    train_docs, train_attn_masks, train_labels = process_data(data_dir=args.data_dir, 
                                                              test_year=args.test_year, 
                                                              max_sentences=args.max_sentences, 
                                                              max_words=args.max_words, 
                                                              tokenizer=tokenizer, 
                                                              training=True)
    print("Processing testing data...")
    test_docs, test_attn_masks, test_labels = process_data(data_dir=args.data_dir, 
                                                           test_year=args.test_year, 
                                                           max_sentences=args.max_sentences, 
                                                           max_words=args.max_words, 
                                                           tokenizer=tokenizer, 
                                                           training=False)

    # Save the processed data as pickle files.
    save_to_hdf5(os.path.join(output_dir, "train_preprocessed.h5"), train_docs, train_attn_masks, train_labels)
    save_to_hdf5(os.path.join(output_dir, "test_preprocessed.h5"), test_docs, test_attn_masks, test_labels)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()