import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
#from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import (
    BATCH_SIZE,
    EPOCHS,
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_NAME,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from evaluate import evaluate_predictions, plot_confusion
from utils import get_device, save_json, save_text


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


@dataclass
class TransformerConfig:
    run_name: str
    freeze_encoder: bool
    model_dir: str
    result_dir: str



def build_model(num_labels: int, freeze_encoder: bool):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if freeze_encoder:
        if hasattr(model, "distilbert"):
            for param in model.distilbert.parameters():
                param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if "classifier" not in name and "pre_classifier" not in name:
                    param.requires_grad = False

    return model



def compute_loss(logits, labels, class_weights, device):
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    return loss_fn(logits, labels)



def run_epoch(model, data_loader, optimizer, scheduler, class_weights, device, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    losses = []
    all_preds = []
    all_labels = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train_mode):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = compute_loss(logits, labels, class_weights, device)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

        preds = torch.argmax(logits, dim=1)
        losses.append(loss.item())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    metrics = evaluate_predictions(all_labels, all_preds)
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    return metrics



def train_transformer(train_df, val_df, test_df, cfg: TransformerConfig):
    device = get_device()
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ReviewDataset(
        train_df["text_transformer"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        MAX_LENGTH,
    )
    val_dataset = ReviewDataset(
        val_df["text_transformer"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        MAX_LENGTH,
    )
    test_dataset = ReviewDataset(
        test_df["text_transformer"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        MAX_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(num_labels=3, freeze_encoder=cfg.freeze_encoder).to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=np.array(train_df["label"].tolist()),
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    best_val_macro_f1 = -1.0
    best_model_path = os.path.join(cfg.model_dir, "best_model.pt")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scheduler, class_weights, device, train_mode=True)
        val_metrics = run_epoch(model, val_loader, optimizer, scheduler, class_weights, device, train_mode=False)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        print(
            f"[{cfg.run_name}] Epoch {epoch}/{EPOCHS} | "
            f"Train Macro-F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_model_path)

    save_json({"history": history}, os.path.join(cfg.result_dir, "training_history.json"))

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_metrics = run_epoch(model, val_loader, optimizer, scheduler, class_weights, device, train_mode=False)
    test_metrics = run_epoch(model, test_loader, optimizer, scheduler, class_weights, device, train_mode=False)

    save_json(val_metrics, os.path.join(cfg.result_dir, "val_metrics.json"))
    save_json(test_metrics, os.path.join(cfg.result_dir, "test_metrics.json"))
    save_text(val_metrics["classification_report_text"], os.path.join(cfg.result_dir, "val_report.txt"))
    save_text(test_metrics["classification_report_text"], os.path.join(cfg.result_dir, "test_report.txt"))

    labels = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    plot_confusion(
        val_metrics["confusion_matrix"],
        labels,
        f"{cfg.run_name} Validation Confusion Matrix",
        os.path.join(cfg.result_dir, "val_confusion_matrix.png"),
    )
    plot_confusion(
        test_metrics["confusion_matrix"],
        labels,
        f"{cfg.run_name} Test Confusion Matrix",
        os.path.join(cfg.result_dir, "test_confusion_matrix.png"),
    )

    tokenizer.save_pretrained(cfg.model_dir)
    model.save_pretrained(cfg.model_dir)

    return {
        "model": cfg.run_name,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }