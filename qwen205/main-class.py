import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import wandb
from tqdm import tqdm
from metric import metric_crossentropy
from utils import load_from_hdf5, risk_metric_map

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=7000):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_year", default=2024, type=int)
    parser.add_argument("--risk_metric", choices=["std", "skew", "kurt", "sortino"], default="std")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--gpu", default="cuda:7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--project", type=str, default="XRC")
    args = parser.parse_args()

    wandb.init(
        name="Qwen2_0dot5B" + f"_{args.test_year}" + f"_S{args.seed}",
        project=args.project + f"_{args.risk_metric}",
        notes="Fine-tune head on Qwen2_0dot5B with per-epoch eval",
    )

    wandb.define_metric("val_f1_macro", summary="max")  # Track highest F1 Macro
    wandb.define_metric("val_spearman_rho", summary="max")  # Track highest Spearman's rho
    wandb.define_metric("val_kendall_tau", summary="max")   # Track highest Kendall's tau

    wandb.config.update(vars(args))

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    #print(f"Device: {device}  —  Test Year {args.test_year}, Risk {args.risk_metric}")

    # load data
    train_docs, train_labels_all = load_from_hdf5(f"../processed/{args.test_year}/prompt/train_preprocessed.h5")
    test_docs,  test_labels_all  = load_from_hdf5(f"../processed/{args.test_year}/prompt/test_preprocessed.h5")

    # pick the one risk metric
    train_labels = train_labels_all[:, risk_metric_map[args.risk_metric]]
    test_labels  = test_labels_all[:,  risk_metric_map[args.risk_metric]]

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        return_dict_in_generate=True,
        output_hidden_states=True # <-- make sure hidden-states are returned
    )
    model.to(device)
    model.eval()

    # freeze all llama parameters
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    num_classes = 3  # labels 0,1,2
    classifier = nn.Linear(hidden_size, num_classes).to(device)
    classifier = classifier.to(device)

    # loss & optimizer (only classifier params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)

    # datasets & loaders
    train_ds = TextDataset(train_docs, train_labels, tokenizer)
    test_ds  = TextDataset(test_docs,  test_labels,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # last hidden state: (batch, seq_len, hidden_size)
                last_hid = outputs.hidden_states[-1]

            # mean‐pool over all tokens
            emb = last_hid.mean(dim=1)           # (batch, hidden_size)
            logits = classifier(emb.to(torch.float32))             # (batch, num_classes)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ─────── VALIDATION ───────
        classifier.eval()
        y_true, y_pred = [], []
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch} — Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                last_hid = outputs.hidden_states[-1]
                emb = last_hid.mean(dim=1)
                logits = classifier(emb.to(torch.float32))
                loss = criterion(logits, labels)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(logits.cpu().numpy())
                total_loss += loss.item()

        avg_val_loss = total_loss / len(test_loader)
        val_acc, val_f1_macro, _, val_spearman_rho, val_kendall_tau = metric_crossentropy(y_pred, y_true)
            
        wandb.log({"train_loss": avg_train_loss, 
                   "val_loss": avg_val_loss,
                   "val_acc": val_acc,
                   "val_f1_macro": val_f1_macro,
                   "val_spearman_rho": val_spearman_rho,
                   "val_kendall_tau": val_kendall_tau,
                   "epoch": epoch})

if __name__ == "__main__":
    main()
