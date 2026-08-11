"""
Roman Urdu Multi-Task NLP Dataset — Emotion Classification (multi-label)

Data: 3,514 comments, 6 labels (joy/anger/sadness/fear/surprise/none),
a comment can have multiple emotions simultaneously.
Result: TF-IDF baseline macro F1 = 0.46 | Transformer macro F1 = 0.40

FINDING: the transformer UNDERPERFORMED the simple baseline here, more so
than in hate speech. Validation loss bottomed out at epoch 3 then rose
while training loss kept falling — a clear overfitting signature, likely
due to the smaller effective per-label sample size in a multi-label setup
with 2,811 training examples. See ../docs/Project_Report.pdf.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset

EMOTION_LABELS = ["joy", "anger", "sadness", "fear", "surprise", "none"]

# --- Build dataset ---
gold_df = pd.read_csv("../dataset/gold_labeled_final.csv")
silver_df = pd.read_csv("../dataset/silver_labeled.csv")

gold_emo = gold_df[["comment_text", "emotion"]].copy()
silver_emo = silver_df[["comment_text", "emotion"]].copy()
emotion_df = pd.concat([gold_emo, silver_emo], ignore_index=True)
emotion_df = emotion_df.dropna(subset=["emotion"])
emotion_df = emotion_df[emotion_df["emotion"] != ""]

for label in EMOTION_LABELS:
    emotion_df[label] = emotion_df["emotion"].apply(lambda x: 1 if label in str(x).split("|") else 0)

print("Total emotion training data:", len(emotion_df))  # 3,514
# Label frequency: joy 43.1%, anger 30.8%, sadness 12.7%, fear 1.7% (thin), surprise 10.5%, none 19.3%

# ===========================================================================
# BASELINE — TF-IDF + multi-label Logistic Regression
# ===========================================================================
X = emotion_df["comment_text"]
y = emotion_df[EMOTION_LABELS]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

baseline_clf = MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
baseline_clf.fit(X_train_vec, y_train)
y_pred = baseline_clf.predict(X_test_vec)
print("\n--- TF-IDF Baseline ---")
print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS, zero_division=0))
# Result: macro F1 = 0.46 (fear's 0.35 F1 is not reliable — only 12 test examples)

# ===========================================================================
# TRANSFORMER — XLM-RoBERTa, multi-label
# ===========================================================================
train_df, test_df = train_test_split(emotion_df, test_size=0.2, random_state=42)

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    enc = tokenizer(batch["comment_text"], truncation=True, padding="max_length", max_length=128)
    enc["labels"] = [[float(batch[l][i]) for l in EMOTION_LABELS] for i in range(len(batch["comment_text"]))]
    return enc

train_ds = Dataset.from_pandas(train_df[["comment_text"] + EMOTION_LABELS]).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df[["comment_text"] + EMOTION_LABELS]).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(EMOTION_LABELS), problem_type="multi_label_classification"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    report = classification_report(labels, preds, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)
    return {"macro_f1": report["macro avg"]["f1-score"], "micro_f1": report["micro avg"]["f1-score"]}

training_args = TrainingArguments(
    output_dir="./emotion_model",
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
    logging_steps=20,
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
# Result: best checkpoint macro F1 = 0.40 (epoch 4) — below the 0.46 TF-IDF baseline

model.save_pretrained("./emotion_final_model")
tokenizer.save_pretrained("./emotion_final_model")
