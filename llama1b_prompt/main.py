import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import argparse
from metric import metric
from utils import collate_fn, load_from_hdf5, risk_metric_map
from sklearn.metrics import classification_report, f1_score

def main():

    parser = argparse.ArgumentParser()

    # General Parameters
    parser.add_argument("--test_year", default="2024", type=int)
    parser.add_argument("--risk_metric", choices=["std", "skew", "kurt", "sortino"], default="std", type=str)
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-1B-Instruct", type=str)
    parser.add_argument("--gpu", default="cuda:0", type=str)
    parser.add_argument("--epochs", default=30, type=int)

    # Others
    parser.add_argument("--project", type=str, default="XRC")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    wandb.init(
        name="llama1B" + f"_{args.test_year}" + f"_S{args.seed}",
        project=args.project + f"_{args.risk_metric}",
        notes="Llama-3.2-1B",
        mode="disabled",
    )
    wandb.config.update(args)

    train_docs, train_labels_all_risks = load_from_hdf5(f"../processed/{args.test_year}/prompt/train_preprocessed.h5")
    test_docs, test_labels_all_risks = load_from_hdf5(f"../processed/{args.test_year}/prompt/test_preprocessed.h5")

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ───────────── In-Context One-Shot Setup ─────────────
    # (1) load the LM for generation
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=False,           # optional quantization
        trust_remote_code=True
    )
    model.to(args.device)
    model.eval()

    y_true, y_pred = [], []

    train_labels = train_labels_all_risks[:, risk_metric_map[args.risk_metric]]
    test_labels = test_labels_all_risks[:, risk_metric_map[args.risk_metric]]

    for text, true_label in zip(test_docs, test_labels):
        # (2) sample one support example per class
        supports = {}
        for cls in [0, 1, 2]:
            idx = random.choice(np.where(train_labels == cls)[0])
            supports[cls] = train_docs[idx]

        # (3) build your prompt
        prompt = "Classify the following text into labels 0, 1, or 2.\n"
        for cls in [0, 1, 2]:
            prompt += f"Text: \"{supports[cls]}\"\nLabel: {cls}\n\n"
        prompt += f"Text: \"{text}\"\nLabel:"

        # (4) generate the label
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)
        out = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,    # greedy
        )
        # decode only the newly generated tokens
        gen = tokenizer.decode(out[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        # take first token and parse int
        pred = None
        if gen:
            tok = gen.split()[0]
            try:
                pred = int(tok)
            except:
                pred = 0
        else:
            pred = 0

        y_true.append(int(true_label))
        y_pred.append(pred)

    # (5) final evaluation
    from sklearn.metrics import classification_report, f1_score
    print(classification_report(y_true, y_pred, digits=4))
    print(f"weighted F1 = {f1_score(y_true, y_pred, average='macro'):.4f}")


if __name__ == "__main__":
    main()
