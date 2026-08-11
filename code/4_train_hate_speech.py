"""
Roman Urdu Multi-Task NLP Dataset — Hate Speech Classification

Data: 3,514 comments (600 gold + 2,914 silver), 3 classes (none/offensive/hate)
Result: TF-IDF baseline macro F1 = 0.57 | Transformer macro F1 = 0.56 (tied)

FINDING: the fine-tuned transformer did not outperform a simple TF-IDF
baseline on this task. See ../docs/Project_Report.pdf for the full
discussion of why (small dataset relative to model capacity).
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset

# --- Build dataset ---
gold_df = pd.read_csv("../dataset/gold_labeled_final.csv")
silver_df = pd.read_csv("../dataset/silver_labeled.csv")

gold_hate = gold_df[["comment_text", "hate_speech"]].copy()
silver_hate = silver_df[["comment_text", "hate_speech"]].copy()
hate_df = pd.concat([gold_hate, silver_hate], ignore_index=True).dropna(subset=["hate_speech"])
print("Total hate speech training data:", len(hate_df))  # 3,514

# ===========================================================================
# BASELINE — TF-IDF + Logistic Regression
# ===========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    hate_df["comment_text"], hate_df["hate_speech"],
    test_size=0.2, random_state=42, stratify=hate_df["hate_speech"]
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

baseline_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
baseline_clf.fit(X_train_vec, y_train)
y_pred = baseline_clf.predict(X_test_vec)
print("\n--- TF-IDF Baseline ---")
print(classification_report(y_test, y_pred))
# Result: macro F1 = 0.57 (majority-baseline macro F1 = 0.29, for comparison)

# ===========================================================================
# TRANSFORMER — XLM-RoBERTa, fine-tuned
# ===========================================================================
label_map = {"none": 0, "offensive": 1, "hate": 2}
hate_df["label"] = hate_df["hate_speech"].map(label_map)
train_df, test_df = train_test_split(hate_df, test_size=0.2, random_state=42, stratify=hate_df["label"])

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["comment_text"], truncation=True, padding="max_length", max_length=128)

train_ds = Dataset.from_pandas(train_df[["comment_text", "label"]]).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df[["comment_text", "label"]]).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Class weights use sqrt scaling, not linear — linear weighting (27x for the
# rarest class) caused training to destabilize and collapse in early
# experiments (macro F1 crashed to 0.11 at one epoch). sqrt scaling fixed this.
counts = Counter(train_df["label"])
total = sum(counts.values())
class_weights = torch.tensor([(total / counts[i]) ** 0.5 for i in range(3)], dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=class_weights.to(model.device))(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    report = classification_report(labels, preds, target_names=["none", "offensive", "hate"], output_dict=True, zero_division=0)
    return {"macro_f1": report["macro avg"]["f1-score"], "accuracy": report["accuracy"]}

training_args = TrainingArguments(
    output_dir="./hate_speech_model",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,               # lower than the 5e-5 default — more stable for this dataset size
    warmup_steps=int(0.1 * (len(train_ds) / 16) * 10),
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=20,
)

trainer = WeightedTrainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
# Result: best checkpoint macro F1 = 0.56 (epoch 3), essentially tied with the TF-IDF baseline

model.save_pretrained("./hate_speech_final_model")
tokenizer.save_pretrained("./hate_speech_final_model")
