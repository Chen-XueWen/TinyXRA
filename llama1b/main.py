import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
from metric import metric_crossentropy
from utils import load_from_hdf5, risk_metric_map

def main():

    parser = argparse.ArgumentParser()

    # General Parameters
    parser.add_argument("--test_year", default="2024", type=int)
    parser.add_argument("--risk_metric", choices=["std", "skew", "kurt", "sortino"], default="std", type=str)
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-1B-Instruct", type=str)
    parser.add_argument("--gpu", default="cuda:7", type=str)
    # Others
    parser.add_argument("--project", type=str, default="XRC")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    print(f"Processing for Test Year {args.test_year}, Risk {args.risk_metric}, Seed {args.seed}")

    train_docs, train_labels_all_risks = load_from_hdf5(f"../processed/{args.test_year}/prompt/train_preprocessed.h5")
    test_docs, test_labels_all_risks = load_from_hdf5(f"../processed/{args.test_year}/prompt/test_preprocessed.h5")

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ───────────── In-Context One-Shot Setup ─────────────
    # (1) load the LM for generation
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    zero_id = tokenizer('0', add_special_tokens=False).input_ids[0]
    one_id = tokenizer('1', add_special_tokens=False).input_ids[0]
    two_id = tokenizer('2', add_special_tokens=False).input_ids[0]

    target_token_ids = [zero_id, one_id, two_id]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        load_in_8bit=False, # optional quantization
        trust_remote_code=True,
        #device_map="auto",
    )
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    train_labels = train_labels_all_risks[:, risk_metric_map[args.risk_metric]]
    test_labels = test_labels_all_risks[:, risk_metric_map[args.risk_metric]]

    for text, true_label in tqdm(zip(test_docs, test_labels), total=len(test_docs), desc="Evaluating"):
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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        # move inputs to the same device as the model's embeddings
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False, # greedy
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )
        # decode only the newly generated tokens
        #gen = tokenizer.decode(out.sequences[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        selected_logits = out.scores[0][0][target_token_ids]

        y_true.append(int(true_label))
        y_pred.append(selected_logits.cpu().numpy())

    # (5) final evaluation
    acc, f1_macro, _, spearman_rho, kendall_tau = metric_crossentropy(y_pred, y_true)
    print("F1 Macro, Spearman Rho, Kendall_Tau")
    print(f1_macro)
    print(spearman_rho)
    print(kendall_tau)

if __name__ == "__main__":
    main()
