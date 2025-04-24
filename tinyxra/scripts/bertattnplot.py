i = 1500

# Your input
attention = outputs.attentions[-1][i].mean(0).cpu()  # replace with actual attention matrix
input_ids = input_ids_flat[i] # replace with actual input_ids
input_ids_list = input_ids.tolist()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
pad_token_id = tokenizer.pad_token_id

# Decode input_ids to tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

# Filter out [PAD] tokens
filtered_indices = [i for i, token_id in enumerate(input_ids_list) if token_id != pad_token_id]
filtered_tokens = [tokens[i] for i in filtered_indices]

# Apply filtering to the attention matrix
filtered_attention = attention[filtered_indices, :][:, filtered_indices]

# Plot and save heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(filtered_attention, xticklabels=filtered_tokens, yticklabels=filtered_tokens, cmap="cividis", cbar=True)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig("attention_no_pad.png", dpi=300)
plt.close()