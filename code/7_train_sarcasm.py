"""
Roman Urdu Multi-Task NLP Dataset — Sarcasm Classification (GOLD-ONLY)

Data: 600 gold, human-labeled comments ONLY (480 train / 120 test).
Sarcasm is deliberately excluded from the silver (LLM-labeled) set — see
3_gold_silver_split_and_labeling.py docstring for the validation experiment
that led to this decision (all 3 tested LLM configurations failed to beat
the 80% majority-class baseline on sarcasm agreement with human labels).

Result: TF-IDF baseline macro F1 = 0.59 | Transformer macro F1 = 0.44
        (majority baseline macro F1 = 0.44)

FINDING: with only 480 training examples, the transformer completely
collapsed to the trivial "always predict not-sarcastic" solution (macro F1
identical to the majority baseline across all 3 epochs before early
stopping). The small TF-IDF model, in contrast, extracted real signal.
This is the clearest illustration in the project of model complexity
requiring a minimum data threshold to be useful at all in this setting.
Notably, this domain-specific model — trained on just 480 real examples —
outperformed every zero-shot LLM tested (see stage 3), suggesting sarcasm
in this dataset is highly context/dataset-specific rather than a general
capability gap.
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

# --- Build dataset (gold only) ---
gold_df = pd.read_csv("../dataset/gold_labeled_final.csv")
sarcasm_df = gold_df[["comment_text", "sarcasm"]].copy().dropna(subset=["sarcasm"])
print("Total sarcasm training data:", len(sarcasm_df))  # 600 (480 no / 120 yes)

# ===========================================================================
# BASELINE — TF-IDF + Logistic Regression
# ===========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    sarcasm_df["comment_text"], sarcasm_df["sarcasm"],
    test_size=0.2, random_state=42, stratify=sarcasm_df["sarcasm"]
)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

baseline_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
baseline_clf.fit(X_train_vec, y_train)
y_pred = baseline_clf.predict(X_test_vec)
print("\n--- TF-IDF Baseline ---")
print(classification_report(y_test, y_pred))
# Result: macro F1 = 0.59 (majority baseline macro F1 = 0.44)

# ===========================================================================
# TRANSFORMER — XLM-RoBERTa (for completeness; expected to underperform
# given the small dataset size, confirmed below)
# ===========================================================================
label_map = {"no": 0, "yes": 1}
sarcasm_df["label"] = sarcasm_df["sarcasm"].map(label_map)
train_df, test_df = train_test_split(sarcasm_df, test_size=0.2, random_state=42, stratify=sarcasm_df["label"])

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["comment_text"], truncation=True, padding="max_length", max_length=128)

train_ds = Dataset.from_pandas(train_df[["comment_text", "label"]]).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df[["comment_text", "label"]]).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

counts = Counter(train_df["label"])
total = sum(counts.values())
class_weights = torch.tensor([(total / counts[i]) ** 0.5 for i in range(2)], dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=class_weights.to(model.device))(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    report = classification_report(labels, preds, target_names=["no", "yes"], output_dict=True, zero_division=0)
    return {"macro_f1": report["macro avg"]["f1-score"], "accuracy": report["accuracy"]}

training_args = TrainingArguments(
    output_dir="./sarcasm_model",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=int(0.1 * (len(train_ds) / 16) * 10),
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=10,
)

trainer = WeightedTrainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
# Result: macro F1 = 0.4444 across ALL epochs before early stopping — the
# model fully collapsed to the majority-class solution and never recovered.
# Recommended model for this task: the TF-IDF baseline (0.59), not the transformer.

model.save_pretrained("./sarcasm_final_model")
tokenizer.save_pretrained("./sarcasm_final_model")
