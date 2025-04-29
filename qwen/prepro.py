import argparse
import os
import h5py
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer

def save_to_hdf5(file_path, docs, attn_masks, labels):
    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("docs", data=np.array(docs, dtype=np.int32), compression="gzip")
        hf.create_dataset("attn_masks", data=np.array(attn_masks, dtype=np.int32))
        hf.create_dataset("labels", data=np.array(labels, dtype=np.int32))
    print(f"Data saved to {file_path}.")

def preprocess_text(text, max_context, tokenizer):
    """
    For Generative Model such as Qwen and Llama, we will have max_context of 350 x 40, 
    which is similar to 350 sentences and 40 words per sentence in the Hierarchical Model.
    """
    # Use the AutoTokenizer to encode the entire text.
    encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_context,
                truncation=True,
                padding='max_length'
            )
    token_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']

    return token_ids, attention_masks

def process_data(data_dir, test_year, max_context, tokenizer, training):
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
            processed, attention_mask = preprocess_text(mdna, max_context, tokenizer)
            processed_docs.append(processed)
            attention_masks.append(attention_mask)
            labels.append((int(std), int(skew), int(kurt), int(sortino)))

    return processed_docs, attention_masks, labels


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
        "--max_context",
        type=int,
        default=350*40,
        help="Maximum number of Tokens for Generative Model, 350 * 40 for fair comparison to TinyXRA."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Model name for AutoTokenizer (default: TinyBERT_General_4L_312D)."
    )
    args = parser.parse_args()
    output_dir = f"../processed/{args.test_year}/{args.model_name}"

    # Check if output_dir exists, else create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Processing training data...")
    train_docs, train_attn_masks, train_labels = process_data(data_dir=args.data_dir, 
                                                              test_year=args.test_year, 
                                                              max_context=args.max_context,
                                                              tokenizer=tokenizer, 
                                                              training=True)
    print("Processing testing data...")
    test_docs, test_attn_masks, test_labels = process_data(data_dir=args.data_dir, 
                                                           test_year=args.test_year, 
                                                           max_context=args.max_context,
                                                           tokenizer=tokenizer, 
                                                           training=False)

    save_to_hdf5(os.path.join(output_dir, "train_preprocessed.h5"), train_docs, train_attn_masks, train_labels)
    save_to_hdf5(os.path.join(output_dir, "test_preprocessed.h5"), test_docs, test_attn_masks, test_labels)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()