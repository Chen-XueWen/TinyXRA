import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import itertools
import h5py
import os
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS


def collate_fn(batch):

    docs = np.array([ele[0] for ele in batch])
    attention_masks = np.array([ele[1] for ele in batch])
    tfidf_weights = np.array([ele[2] for ele in batch])
    labels = np.array([ele[3] for ele in batch])

    docs_tensor = torch.tensor(docs, dtype=torch.long)
    attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long)
    tfidf_weights_tensor = torch.tensor(tfidf_weights, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    output = {"input_ids": docs_tensor,
              "attention_masks": attention_masks_tensor,
              "tfidf_weights": tfidf_weights_tensor,
              "targets": labels_tensor,
              }
    
    return output

def load_from_hdf5(file_path):
    """
    Loads processed data from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        np.array: Loaded documents.
        np.array: Loaded labels.
    """
    with h5py.File(file_path, "r") as hf:
        docs = hf["docs"][:]
        attn_masks = hf["attn_masks"][:]
        tfidf_weights = hf["tfidf_weights"][:]
        labels = hf["labels"][:]
    return docs, attn_masks, tfidf_weights, labels

def get_triplets(embeddings, labels):
    """
    embeddings: (B, embed_dim)
    labels: (B,)  with values in [0,1,2]
    Returns anchor, positive, negative
    Each is of shape (N, embed_dim), where N <= B
    """
    device = embeddings.device
    anchor_list = []
    positive_list = []
    negative_list = []
    
    # Convert labels to CPU for indexing
    labels_cpu = labels.cpu()
    
    for i in range(len(labels)):
        anchor_label = labels_cpu[i].item()
        
        # Find indices of positives and negatives
        positive_indices = (labels_cpu == anchor_label).nonzero(as_tuple=True)[0]
        negative_indices = (labels_cpu != anchor_label).nonzero(as_tuple=True)[0]
        
        # Remove the anchor itself from the positive candidates
        positive_indices = positive_indices[positive_indices != i]
        
        # We need at least 1 positive and 1 negative
        if len(positive_indices) < 1 or len(negative_indices) < 1:
            continue
        
        # Randomly select 1 positive and 1 negative
        pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
        neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
        
        anchor_list.append(embeddings[i].unsqueeze(0))
        positive_list.append(embeddings[pos_idx])
        negative_list.append(embeddings[neg_idx])
    
    if len(anchor_list) == 0:
        # In a worst-case scenario (e.g., batch too small or all same label),
        # fallback to returning None or zero-length.
        return None, None, None
    
    anchor_tensor = torch.cat(anchor_list, dim=0).to(device)      # (N, embed_dim)
    positive_tensor = torch.cat(positive_list, dim=0).to(device)  # (N, embed_dim)
    negative_tensor = torch.cat(negative_list, dim=0).to(device)  # (N, embed_dim)
    
    return anchor_tensor, positive_tensor, negative_tensor



def triplet_ranking_loss(outputs, targets, margin=0.1):
    """
    Compute triplet loss using **all possible triplets** where:
    - Anchor (A) is always from **Medium Risk (1)**
    - Positive (P) is always from **High Risk (2)**
    - Negative (N) is always from **Low Risk (0)**

    :param outputs: Model logits (ranking scores), shape (batch_size, 1)
    :param targets: True risk labels, shape (batch_size,)
    :param margin: Triplet loss margin
    """
    outputs = outputs.squeeze()  # Ensure shape (batch_size,)
    loss = 0
    num_triplets = 0

    # Get indices for each risk level
    anchor_indices = [i for i in range(len(outputs)) if targets[i] == 1]  # Medium risk
    positive_indices = [i for i in range(len(outputs)) if targets[i] == 2]  # High risk
    negative_indices = [i for i in range(len(outputs)) if targets[i] == 0]  # Low risk

    # Generate all valid triplets
    triplets = itertools.product(anchor_indices, positive_indices, negative_indices)

    for anchor_idx, positive_idx, negative_idx in triplets:
        anchor = outputs[anchor_idx]
        positive = outputs[positive_idx]
        negative = outputs[negative_idx]

        # Standard triplet loss
        triplet_loss = F.relu(anchor - positive + margin) + F.relu(negative - anchor + margin)
        loss += triplet_loss
        num_triplets += 1

    return loss / num_triplets if num_triplets > 0 else torch.tensor(0.0, device=outputs.device)


def plot_all_top5_word_attention_heatmaps(
    input_ids, sent_attns, word_attns, tokenizer, year, risk_metric, 
    all_preds, all_targets, output_dir="attention_heatmaps"
):
    # Define punctuation set and stopwords.
    punct_set = set(string.punctuation)
    stopwords_set = STOPWORDS.union({"the", "and", "is", "are", "i", "[CLS]", "[SEP]", "[PAD]", "[UNK]"})

    folderpath = os.path.join(output_dir, f"{risk_metric}/{year}")
    os.makedirs(folderpath, exist_ok=True)
    batch_size = input_ids.shape[0]

    pred_scores = np.array(all_preds)
    true_labels = np.array(all_targets)

    # Compute thresholds based on percentiles (adaptive binning)
    threshold_1 = np.percentile(pred_scores, 30)  # First threshold (30th percentile)
    threshold_2 = np.percentile(pred_scores, 70)  # Second threshold (70th percentile)

    # Convert continuous scores to discrete labels
    pred_labels = np.digitize(pred_scores, bins=[threshold_1, threshold_2])

    for sample_idx in range(batch_size):
        input_ids_sample = input_ids[sample_idx]      # (350, 40)
        word_attns_sample = word_attns[sample_idx]    # (350, 40)
        sent_attns_sample = sent_attns[sample_idx]    # (350,)
        pred_label_sample = int(pred_labels[sample_idx])
        true_label_sample = int(true_labels[sample_idx])
        pred_score_sample = round(float(pred_scores[sample_idx]), 4)

        # Get indices of top 5 sentences by attention
        top5_indices = torch.topk(sent_attns_sample, k=5).indices.cpu()

        # Subset the tokens and attention for top 5 sentences
        selected_input_ids = input_ids_sample[top5_indices]    # (5, 40)
        selected_word_attns = word_attns_sample[top5_indices]  # (5, 40)

        token_grid = []
        attn_grid = []
        for i, sentence in enumerate(selected_input_ids):
            tokens = tokenizer.convert_ids_to_tokens(sentence)
            token_filtered = []
            attn_filtered = []
            for j, token in enumerate(tokens):
                if token in stopwords_set:
                    continue
                # Skip tokens that are solely punctuation.
                if all(ch in punct_set for ch in token):
                    continue
                # Remove internal punctuation and trim
                token = re.sub(r"[^\w\s]", "", token).strip()
                if not token:
                    continue

                token_filtered.append(token)
                attn_filtered.append(selected_word_attns[i][j].item())
            token_grid.append(token_filtered)
            attn_grid.append(attn_filtered)
        
        ### Padding for token and attn grid
        max_length = max([len(sent) for sent in token_grid])

        for i in range(len(token_grid)):
            token_pad_length = max_length - len(token_grid[i])
            attn_pad_length = max_length - len(attn_grid[i])

            # Pad tokens with "" or "[PAD]"
            token_grid[i] += ["[PAD]"] * token_pad_length

            # Pad attention with 0.0
            attn_grid[i] += [0.0] * attn_pad_length

        attn_grid = np.array(attn_grid)
        if attn_grid.shape[1] == 0:
            print(f"Skipping heatmap for Doc {sample_idx} Year {year} Risk {risk_metric}, empty attn_grid")
            continue
        
        # Make the figure larger if you need more space for vertical text
        plt.figure(figsize=(20, 8))
        ax = sns.heatmap(attn_grid, cmap="cividis", cbar=True, linewidths=0.5, linecolor='gray', annot=False)
        # Annotate with tokens (vertical text, larger font)
        all_attn_values = [val for row in attn_grid for val in row]
        threshold = np.percentile(all_attn_values, 99)  # Top 1%
        for i in range(attn_grid.shape[0]):  # 5 rows (top sentences)
            for j in range(attn_grid.shape[1]):  # 40 columns (words)
                token = token_grid[i][j]
                if token != "[PAD]":
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        token,
                        ha='center',
                        va='center',
                        rotation=90,
                        fontsize=12,
                        color='white' if attn_grid[i][j] < threshold else 'black'
                    )

        # Optional: label Y-axis with original sentence indices
        y_labels = [f"Sent {idx.item()}" for idx in top5_indices]
        ax.set_yticklabels(y_labels, rotation=0)

        ax.set_title(
            f"Test Year {year}: Document {sample_idx} - Top 5 Sentence Word Attention, "
            f"Gold Bin = {true_label_sample}, Pred Bin = {pred_label_sample}, Pred Score = {pred_score_sample}",
            fontsize=14
        )
        ax.set_xlabel("Word Position")
        ax.set_ylabel("Top Sentences")
        plt.tight_layout()

        filename = f"Doc_{sample_idx}_top5_word_attention.png"
        filepath = os.path.join(folderpath, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()


def generate_word_cloud(input_ids, sent_attns, word_attns, tokenizer, year, risk_metric, 
                        all_preds, top_percent=0.3, output_dir="attention_heatmaps"):
    """
    Generate word clouds for groups of documents with discrete prediction labels.
    For overlapping tokens across groups, the new frequency for each token is computed as:
        new_freq = freq_in_this_group - max(freq_in_other_groups).
    If new_freq <= 0, the token is dropped from that group.
    """
    # Define punctuation set and stopwords.
    punct_set = set(string.punctuation)
    stopwords_set = STOPWORDS.union({"the", "and", "is", "are", "i", "[CLS]", "[SEP]", "[PAD]", "[UNK]"})
    
    # Process prediction scores into discrete labels (0, 1, or 2)
    pred_scores = np.array(all_preds)
    threshold_1 = np.percentile(pred_scores, 30)  # 30th percentile
    threshold_2 = np.percentile(pred_scores, 70)  # 70th percentile
    pred_labels = np.digitize(pred_scores, bins=[threshold_1, threshold_2]).squeeze()  # shape: (num_docs,)

    num_docs, num_sentences, num_words = input_ids.shape
    # Convert input_ids to tokens for all documents.
    all_tokens = []
    for doc_idx in range(num_docs):
        doc_tokens = []
        for sent_idx in range(num_sentences):
            token_ids = input_ids[doc_idx, sent_idx].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            # Remove subword markers if desired.
            cleaned_tokens = [tok.replace("##", "") for tok in tokens]
            doc_tokens.append(cleaned_tokens)
        all_tokens.append(doc_tokens)
    
    unique_labels = np.unique(pred_labels)

    # Accumulate word frequency dictionaries per group.
    label_word_freq = {}
    
    for label in unique_labels:
        # Get document indices for this label.
        doc_indices = np.where(pred_labels == label)[0]
        if len(doc_indices) == 0:
            continue
        
        # Select attentions and tokens.
        group_word_attns = word_attns[doc_indices]    # shape: [num_group_docs, num_sentences, num_words]
        group_sent_attns = sent_attns[doc_indices]      # shape: [num_group_docs, num_sentences]
        group_tokens = [all_tokens[i] for i in doc_indices]  # nested list for docs in group
        
        # Compute weighted importance.
        weighted_importance = group_word_attns * group_sent_attns.unsqueeze(-1)
        
        # Accumulate weighted scores into a frequency dictionary.
        word_frequencies = defaultdict(float)
        grp_docs, grp_sents, grp_words = weighted_importance.shape
        
        # Loop over each document in the group for per-document threshold computation.
        for d in range(grp_docs):
            # Extract weighted importance for one document (shape: [num_sentences, num_words])
            weighted_doc = weighted_importance[d]
            # Flatten to compute the quantile threshold for this document
            flattened_doc = weighted_doc.view(-1)
            quantile_threshold_doc = torch.quantile(flattened_doc, q=1 - top_percent)
            # Create a boolean mask for tokens in this document above the threshold
            mask_doc = (weighted_doc >= quantile_threshold_doc)
        
            # Loop through sentences and words in the document using the document-specific mask.
            for s in range(grp_sents):
                for w in range(grp_words):
                    if mask_doc[s, w]:
                        token = group_tokens[d][s][w]
                        # Skip tokens that are in the stopwords set.
                        if token in stopwords_set:
                            continue
                        # Skip tokens that are solely punctuation.
                        if all(ch in punct_set for ch in token):
                            continue
                        # Remove internal punctuation and trim
                        token = re.sub(r"[^\w\s]", "", token).strip()
                        if not token:
                            continue
                        
                        score = weighted_doc[s, w].item()
                        word_frequencies[token] += score
        label_word_freq[label] = word_frequencies

    # Compute the union of all tokens across groups.
    union_tokens = set()
    for freq in label_word_freq.values():
        union_tokens.update(freq.keys())
    
    # For each token that appears in multiple groups, adjust its frequency by taking the difference.
    for token in union_tokens:
        token_freq = {}
        for label, freq in label_word_freq.items():
            if token in freq:
                token_freq[label] = freq[token]
        # Only adjust if the token appears in more than one group.
        if len(token_freq) > 1:
            for label, original_freq in token_freq.items():
                # Get the maximum frequency of this token among the other groups.
                other_freqs = [f for lab, f in token_freq.items() if lab != label]
                other_max = max(other_freqs) if other_freqs else 0
                new_freq = original_freq - other_max
                if new_freq <= 0:
                    del label_word_freq[label][token]
                else:
                    label_word_freq[label][token] = new_freq

    # Prepare output folder.
    folderpath = os.path.join(output_dir, f"{risk_metric}", f"{year}")
    os.makedirs(folderpath, exist_ok=True)
    
    # Generate and save the word cloud for each group.
    cloud_width = 800
    cloud_height = 400
    
    for label, word_frequencies in label_word_freq.items():
        wc = WordCloud(
            width=cloud_width,
            height=cloud_height,
            background_color="white",
            max_words=200
        ).generate_from_frequencies(dict(word_frequencies))
        
        plt.figure(figsize=(cloud_width/100, cloud_height/100))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Clouds for Predicted Label {label} Across All Documents")
        filename = f"cloud_label_{label}.png"
        filepath = os.path.join(folderpath, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved word cloud for label {label} at {filepath}")


risk_metric_map = {
    "std":0, 
    "skew":1, 
    "kurt":2, 
    "sortino":3
}