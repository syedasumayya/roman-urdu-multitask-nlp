"""
Roman Urdu Multi-Task NLP Dataset — Joint Multi-Task Model

The core novel experiment of this project: does jointly training hate
speech, emotion, and code-switch tagging on ONE shared encoder help or
hurt compared to training each separately?

Data: 3,513 comments with all 3 task labels present and aligned.
(Sarcasm and misinformation are excluded — see individual task scripts
for why each is handled separately.)

Architecture: one shared XLM-RoBERTa encoder + 3 small task-specific heads
(hate: 3-class, emotion: 6-label multi-label, code-switch: per-token 3-class),
trained jointly with a combined loss.

===========================================================================
RESULTS — three configurations tested:
===========================================================================
                          Single-task   Multi-task    Multi-task
                          (best)        (equal wt.)   (reweighted)
Hate speech F1            0.57          0.55          0.53
Emotion F1                0.46          0.28          0.31
Code-switch tag F1        0.80          0.78          0.66
Average F1                 --           0.535         0.501

FINDING 1 (equal-weight run): negative transfer. Code-switch tagging has
far more effective training signal (each comment yields ~13 word-level
labels vs. 1 hate label + 1 emotion vector), so its gradients dominated
the shared encoder during joint training, degrading the other two tasks
-- especially emotion (-18 F1 points vs. its single-task score).

FINDING 2 (reweighted run, HATE_WEIGHT=2.0, EMOTION_WEIGHT=3.0,
TAG_WEIGHT=0.3): down-weighting the dominant task partially recovered
emotion (0.28 -> 0.31) but degraded code-switch tagging MORE than
proportionally (0.78 -> 0.66), for a LOWER overall average than the naive
equal-weighted run. This is a genuine Pareto trade-off, not a fix -- loss
reweighting alone does not resolve the negative transfer; it just
redistributes it. More sophisticated multi-task techniques (e.g.
gradient-based task balancing) or more training data per task would
likely be needed. See ../docs/Project_Report.pdf for full discussion.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset

# ---------------------------------------------------------------------------
# Shared setup — data loading, common to both configurations below
# ---------------------------------------------------------------------------
def parse_tagged(tagged_str):
    if pd.isna(tagged_str) or tagged_str == "":
        return None
    pairs = tagged_str.split(" ")
    words, tags = [], []
    for p in pairs:
        idx = p.rfind("/")
        if idx == -1:
            continue
        words.append(p[:idx])
        tags.append(p[idx + 1:])
    return words, tags

EMOTION_LABELS = ["joy", "anger", "sadness", "fear", "surprise", "none"]
HATE_LABELS = {"none": 0, "offensive": 1, "hate": 2}
TAG_LABELS = {"urdu": 0, "english": 1, "ambiguous": 2}

gold_df = pd.read_csv("../dataset/gold_labeled_final.csv")
silver_df = pd.read_csv("../dataset/silver_labeled.csv")

records = []
for df, source in [(gold_df, "gold"), (silver_df, "silver")]:
    for _, row in df.iterrows():
        parsed = parse_tagged(row.get("code_switch_tags"))
        if not parsed:
            continue
        words, tags = parsed
        if len(words) != len(tags) or len(words) == 0:
            continue
        if any(t not in TAG_LABELS for t in tags):
            continue
        if pd.isna(row.get("hate_speech")) or row["hate_speech"] not in HATE_LABELS:
            continue
        if pd.isna(row.get("emotion")) or row["emotion"] == "":
            continue
        emo_vec = [1.0 if lbl in str(row["emotion"]).split("|") else 0.0 for lbl in EMOTION_LABELS]
        records.append({
            "words": words, "hate_label": HATE_LABELS[row["hate_speech"]],
            "emotion_labels": emo_vec, "tag_labels": [TAG_LABELS[t] for t in tags],
        })

print("Total examples with all 3 task labels present:", len(records))  # 3,513

train_records, test_records = train_test_split(records, test_size=0.2, random_state=42)

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def build_dataset(recs):
    return Dataset.from_dict({
        "tokens": [r["words"] for r in recs], "hate_label": [r["hate_label"] for r in recs],
        "emotion_labels": [r["emotion_labels"] for r in recs], "tags": [r["tag_labels"] for r in recs],
    })

def tokenize_and_align(batch):
    tokenized = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128)
    all_tag_labels = []
    for i, tags in enumerate(batch["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word_id:
                label_ids.append(tags[wid])
            else:
                label_ids.append(-100)
            prev_word_id = wid
        all_tag_labels.append(label_ids)
    tokenized["tag_labels"] = all_tag_labels
    tokenized["hate_label"] = batch["hate_label"]
    tokenized["emotion_labels"] = batch["emotion_labels"]
    return tokenized

train_ds = build_dataset(train_records)
test_ds = build_dataset(test_records)
train_ds = train_ds.map(tokenize_and_align, batched=True, remove_columns=train_ds.column_names)
test_ds = test_ds.map(tokenize_and_align, batched=True, remove_columns=test_ds.column_names)

# ---------------------------------------------------------------------------
# Multi-task architecture — one shared encoder, three heads
# ---------------------------------------------------------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_hate=3, num_emotion=6, num_tags=3,
                 hate_weight=1.0, emotion_weight=1.0, tag_weight=1.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.hate_head = nn.Linear(hidden, num_hate)
        self.emotion_head = nn.Linear(hidden, num_emotion)
        self.tag_head = nn.Linear(hidden, num_tags)
        self.hate_weight = hate_weight
        self.emotion_weight = emotion_weight
        self.tag_weight = tag_weight

    def forward(self, input_ids, attention_mask, hate_label=None, emotion_labels=None, tag_labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state       # per-token, for code-switch tagging
        pooled_output = sequence_output[:, 0, :]           # CLS token, for sentence-level tasks

        hate_logits = self.hate_head(pooled_output)
        emotion_logits = self.emotion_head(pooled_output)
        tag_logits = self.tag_head(sequence_output)

        loss = None
        if hate_label is not None:
            hate_loss = nn.CrossEntropyLoss()(hate_logits, hate_label)
            emotion_loss = nn.BCEWithLogitsLoss()(emotion_logits, emotion_labels)
            tag_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                tag_logits.view(-1, tag_logits.shape[-1]), tag_labels.view(-1)
            )
            loss = self.hate_weight * hate_loss + self.emotion_weight * emotion_loss + self.tag_weight * tag_loss

        return {"loss": loss, "hate_logits": hate_logits, "emotion_logits": emotion_logits, "tag_logits": tag_logits}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    hate_logits, emotion_logits, tag_logits = predictions
    hate_labels, emotion_labels, tag_labels = labels

    hate_preds = np.argmax(hate_logits, axis=1)
    hate_f1 = classification_report(hate_labels, hate_preds, output_dict=True, zero_division=0)["macro avg"]["f1-score"]

    emotion_preds = (1 / (1 + np.exp(-emotion_logits)) > 0.5).astype(int)
    emotion_f1 = classification_report(emotion_labels, emotion_preds, output_dict=True, zero_division=0)["macro avg"]["f1-score"]

    tag_preds = np.argmax(tag_logits, axis=2)
    flat_true, flat_pred = [], []
    for pr, lb in zip(tag_preds, tag_labels):
        for p, l in zip(pr, lb):
            if l != -100:
                flat_true.append(l); flat_pred.append(p)
    tag_f1 = classification_report(flat_true, flat_pred, output_dict=True, zero_division=0)["macro avg"]["f1-score"]

    avg_f1 = (hate_f1 + emotion_f1 + tag_f1) / 3
    return {"hate_f1": hate_f1, "emotion_f1": emotion_f1, "tag_f1": tag_f1, "avg_f1": avg_f1}

def run_multitask(hate_weight, emotion_weight, tag_weight, output_dir):
    model = MultiTaskModel(MODEL_NAME, hate_weight=hate_weight, emotion_weight=emotion_weight, tag_weight=tag_weight)
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=10,
        per_device_train_batch_size=16, per_device_eval_batch_size=16,
        learning_rate=2e-5, warmup_steps=int(0.1 * (len(train_ds) / 16) * 10),
        weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="avg_f1",
        logging_steps=20, remove_unused_columns=False,
        label_names=["hate_label", "emotion_labels", "tag_labels"],
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    return model, trainer

# ---------------------------------------------------------------------------
# RUN 1 — equal weighting (naive multi-task baseline)
# ---------------------------------------------------------------------------
model_v1, trainer_v1 = run_multitask(hate_weight=1.0, emotion_weight=1.0, tag_weight=1.0, output_dir="./multitask_model_v1")
# Result: hate 0.547, emotion 0.276, tag 0.782, avg 0.535 (epoch 10, no early stop)
torch.save(model_v1.state_dict(), "multitask_model_v1_final.pt")

# ---------------------------------------------------------------------------
# RUN 2 — reweighted (attempt to counteract negative transfer)
# ---------------------------------------------------------------------------
model_v2, trainer_v2 = run_multitask(hate_weight=2.0, emotion_weight=3.0, tag_weight=0.3, output_dir="./multitask_model_v2")
# Result: hate 0.532, emotion 0.309, tag 0.661, avg 0.501 (epoch 9 best) — a
# real trade-off, not a fix: emotion improved, tagging degraded MORE, net
# average is LOWER than the equal-weighted run.
torch.save(model_v2.state_dict(), "multitask_model_v2_final.pt")

tokenizer.save_pretrained("./multitask_tokenizer")
print("Both multi-task configurations complete. See docstring above for full results.")
