import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.sampler import Sampler
from transformers import get_linear_schedule_with_warmup

from tqdm.auto import tqdm
import wandb
import random
import argparse
from model import HierarchicalNet
from metric import metric
from utils import collate_fn, load_from_hdf5, risk_metric_map, triplet_ranking_loss

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

class StratifiedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        Ensures each batch contains at least 1 sample from each risk level (0,1,2).
        The remaining slots are randomly filled.
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # Store indices per class
        self.indices_per_class = {
            0: [i for i, label in enumerate(dataset.labels) if label == 0],  # Low risk
            1: [i for i, label in enumerate(dataset.labels) if label == 1],  # Medium risk
            2: [i for i, label in enumerate(dataset.labels) if label == 2],  # High risk
        }

        # All dataset indices for random selection
        self.all_indices = list(range(len(dataset)))

    def __iter__(self):
        # Shuffle each class separately at the start of every epoch
        for key in self.indices_per_class:
            random.shuffle(self.indices_per_class[key])
        
        random.shuffle(self.all_indices)

        num_batches = len(self.all_indices) // self.batch_size

        for _ in range(num_batches):
            batch = []

            # Ensure at least 1 sample from each class (if available)
            for key in self.indices_per_class:
                if self.indices_per_class[key]:  # Ensure we have samples
                    batch.append(random.choice(self.indices_per_class[key]))

            # Fill the remaining batch size with random samples (without removal)
            while len(batch) < self.batch_size:
                batch.append(random.choice(self.all_indices))

            yield batch

    def __len__(self):
        """
        Returns the total number of batches.
        """
        return len(self.all_indices) // self.batch_size

class Trainer:
    def __init__(self, args, model, train_dataset, val_dataset):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_save_path = args.model_save_path
        self.best_val_f1 = 0
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder', 'classifier']
        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': args.weight_decay,
                'lr': args.classifier_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': 0.0,
                'lr': args.classifier_lr
            }
        ]
        
        self.optimizer = torch.optim.AdamW(grouped_params)
        # Include Warm-up steps to stabilize early training
        total_steps = len(train_dataset) * args.epochs
        warmup_steps = int(total_steps * 0.1)  # 10% of total steps for warm-up
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)
        self.scaler = GradScaler()
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            # Use autocast for mixed precision
            with autocast():
                outputs, _, _, document_embs = self.model(
                    batch['input_ids'].to(self.args.device), 
                    batch['attention_masks'].to(self.args.device)
                )
                # Compute pairwise ranking loss
                loss = triplet_ranking_loss(outputs, batch['targets'].to(self.args.device))

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        
            self.model.zero_grad()
            self.scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def eval_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                with autocast():
                    outputs, _, _, document_embs = self.model(
                        batch['input_ids'].to(self.args.device), 
                        batch['attention_masks'].to(self.args.device)
                    )
                    # Compute pairwise ranking loss
                    loss = triplet_ranking_loss(outputs, batch['targets'].to(self.args.device))

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch['targets'].cpu().numpy())

        acc, f1_macro, _, spearman_rho, kendall_tau = metric(all_preds, all_targets)
        avg_loss = total_loss / len(dataloader)

        return avg_loss, acc, f1_macro, spearman_rho, kendall_tau

    def train(self):
        for epoch in range(self.args.epochs):
            train_dataloader = self.train_dataset
            val_dataloader = self.val_dataset
            avg_train_loss = self.train_epoch(train_dataloader, epoch)
            avg_val_loss, val_acc, val_f1_macro, val_spearman_rho, val_kendall_tau = self.eval_epoch(val_dataloader)
            wandb.log({"train_loss": avg_train_loss, 
                       "val_loss": avg_val_loss,
                       "val_acc": val_acc,
                       "val_f1_macro": val_f1_macro,
                       "val_spearman_rho": val_spearman_rho,
                       "val_kendall_tau": val_kendall_tau,
                       "epoch": epoch})

            print(f"Epoch {epoch + 1}: Training Loss {avg_train_loss}, Validation Loss {avg_val_loss}")
            
            if val_f1_macro > self.best_val_f1:
                self.best_val_f1 = val_f1_macro
                print(f"New best validation F1 Macro Score {self.best_val_f1}. Saving model and tokenizer.")
                #torch.save(self.model.state_dict(), self.model_save_path)


    def evaluate(self):
        val_dataloader = self.val_dataset
        avg_val_loss, val_acc, val_f1_macro, val_spearman_rho, val_kendall_tau = self.eval_epoch(val_dataloader)
        
        print(f"Validation Loss: {avg_val_loss}")
        print(f"Validation Accuracy: {val_acc}")
        print(f"Validation F1 Macro: {val_f1_macro}")
        print(f"Validation Spearman's Rho: {val_spearman_rho}")
        print(f"Validation Kendall's Tau: {val_kendall_tau}")

def main():

    parser = argparse.ArgumentParser()

    # General Parameters
    parser.add_argument("--test_year", default="2024", type=int)
    parser.add_argument("--risk_metric", choices=["std", "skew", "kurt", "sortino"], default="std", type=str)
    parser.add_argument("--model_name_or_path", default="huawei-noah/TinyBERT_General_4L_312D", type=str)
    parser.add_argument("--gpu", default="cuda:7", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
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
    parser.add_argument("--project", type=str, default="XRC_Ablation2")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    args.model_save_path = f'./best_f1_{args.test_year}_{args.seed}.pth'
    #args.model_load_path = f'./best_f1_{args.data_type}_{args.seed}.pth'
    
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    wandb.init(
        name="MaxPool" + f"_{args.test_year}" + f"_S{args.seed}",
        project=args.project + f"_{args.risk_metric}",
        notes="TinyBERT using CLS and max pooling",
        #mode="disabled",
    )

    wandb.define_metric("val_f1_macro", summary="max")  # Track highest F1 Macro
    wandb.define_metric("val_spearman_rho", summary="max")  # Track highest Spearman's rho
    wandb.define_metric("val_kendall_tau", summary="max")   # Track highest Kendall's tau

    wandb.config.update(args)

    train_docs, train_attn_masks, train_labels = load_from_hdf5(f"../../../processed/{args.test_year}/{args.model_name_or_path}/train_preprocessed.h5")
    test_docs, test_attn_masks, test_labels = load_from_hdf5(f"../../../processed/{args.test_year}/{args.model_name_or_path}/test_preprocessed.h5")

    train_dataset = DocClassificationDataset(train_docs, train_attn_masks, train_labels[:, risk_metric_map[args.risk_metric]])
    test_dataset = DocClassificationDataset(test_docs, test_attn_masks, test_labels[:, risk_metric_map[args.risk_metric]])

    # Define the sampler
    train_sampler = StratifiedBatchSampler(train_dataset, batch_size=args.batch_size)
    
    training_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
    testing_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = HierarchicalNet(args)
    model.to(device)
    
    trainer = Trainer(args=args,
                      model=model,
                      train_dataset=training_dataloader,
                      val_dataset=testing_dataloader)
    
    if args.model_load_path:
        model.load_state_dict(torch.load(args.model_load_path, map_location=device))
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
