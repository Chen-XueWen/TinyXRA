import argparse
import os
import h5py
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer

def save_to_hdf5(file_path, docs, labels):
    with h5py.File(file_path, "w") as hf:
        dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset("docs", data=np.array(docs, dtype=dt), compression="gzip")
        hf.create_dataset("labels", data=np.array(labels, dtype=np.int32))
    print(f"Data saved to {file_path}.")

def preprocess_text(text, max_context):
    """
    For Generative Model such as Qwen and Llama, we will have max_context of 350 x 40, 
    which is similar to 350 sentences and 40 words per sentence in the Hierarchical Model.
    Also to make it more manageable due to limited hardware. 
    For generalizablility, we use sent_tokenizer and word_tokenizer.
    Since generative model will perform the tokenization and the text generation through pipeline approach
    """
    # Use the AutoTokenizer to encode the entire text.
    filtered_sents = " ".join(sent_tokenize(text)[3:]) #First 3 sentences are usually titles
    tokens = word_tokenize(filtered_sents)[:max_context]
    trunc_text = " ".join(tokens)
    return trunc_text

def process_data(data_dir, test_year, max_context, training):
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
            processed = preprocess_text(mdna, max_context)
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
        "--max_context",
        type=int,
        default=350*40,
        help="Maximum number of Tokens for Generative Model, 350 * 40 for fair comparison to TinyXRA."
    )

    args = parser.parse_args()
    output_dir = f"../processed/{args.test_year}/prompt"

    # Check if output_dir exists, else create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    print("Processing training data...")
    train_docs, train_labels = process_data(data_dir=args.data_dir, 
                                            test_year=args.test_year, 
                                            max_context=args.max_context,
                                            training=True)
    print("Processing testing data...")
    test_docs, test_labels = process_data(data_dir=args.data_dir, 
                                          test_year=args.test_year, 
                                          max_context=args.max_context,
                                          training=False)

    save_to_hdf5(os.path.join(output_dir, "train_preprocessed.h5"), train_docs, train_labels)
    save_to_hdf5(os.path.join(output_dir, "test_preprocessed.h5"), test_docs, test_labels)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()