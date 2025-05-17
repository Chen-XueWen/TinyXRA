import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import wandb
import argparse
from metric import metric
from utils import collate_fn, load_from_hdf5, risk_metric_map

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score, classification_report

class DocClassificationDataset(Dataset):
    def __init__(self, docs, attn_masks, labels):
        self.docs = docs
        self.attn_masks = attn_masks
        self.labels = labels

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        attn_mask = self.attn_masks[idx]
        label = self.labels[idx]
        return doc, attn_mask, label

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("targets").long()
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    return {'balanced_accuracy' : balanced_accuracy_score(predictions, labels),
    'accuracy':accuracy_score(predictions,labels)}

def main():

    parser = argparse.ArgumentParser()

    # General Parameters
    parser.add_argument("--test_year", default="2024", type=int)
    parser.add_argument("--risk_metric", choices=["std", "skew", "kurt", "sortino"], default="std", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-3B", type=str)
    parser.add_argument("--gpu", default="cuda:0", type=str) # irrelevant for with accelerate
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=30, type=int)

    # Model Parameters
    parser.add_argument('--word_hidden_size', type=int, default=128)
    parser.add_argument('--sentence_hidden_size', type=int, default=128)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--classifier_lr', type=float, default=6e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=2.5)

    # Others
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--model_load_path", type=str)
    parser.add_argument("--project", type=str, default="XRC")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    args.model_save_path = f'./checkpoints/{args.seed}/{args.risk_metric}_{args.test_year}.pth'
    #args.model_load_path = f'./checkpoints/{args.seed}/{args.risk_metric}_{args.test_year}.pth'
    
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    wandb.init(
        name="Qwen3B" + f"_{args.test_year}" + f"_S{args.seed}",
        project=args.project + f"_{args.risk_metric}",
        notes="Qwen2.5-3B ",
        mode="disabled",
    )
    wandb.config.update(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_docs, train_attn_masks, train_labels = load_from_hdf5(f"../processed/{args.test_year}/{args.model_name_or_path}/train_preprocessed.h5")
    test_docs, test_attn_masks, test_labels = load_from_hdf5(f"../processed/{args.test_year}/{args.model_name_or_path}/test_preprocessed.h5")

    train_dataset = DocClassificationDataset(train_docs, train_attn_masks, train_labels[:, risk_metric_map[args.risk_metric]])
    test_dataset = DocClassificationDataset(test_docs, test_attn_masks, test_labels[:, risk_metric_map[args.risk_metric]])

    # Load Quantized Model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True, 
        bnb_4bit_quant_type = 'nf4',
        bnb_4bit_use_double_quant = True, 
        bnb_4bit_compute_dtype = torch.bfloat16)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        num_labels=3,
        device_map='auto')

    lora_config = LoraConfig(
        r = 4, 
        lora_alpha = 8,
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout = 0.05, 
        bias = 'none',
        task_type = 'SEQ_CLS')
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir = 'checkpoints',
        learning_rate = 1e-5,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        num_train_epochs = 1,
        logging_steps=1,
        weight_decay = 0.01,
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        load_best_model_at_end = True,
        report_to="none")

    trainer = CustomTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        tokenizer = tokenizer,
        data_collator = collate_fn,
        compute_metrics = compute_metrics)

    train_result = trainer.train()


if __name__ == "__main__":
    main()
