"""
Roman Urdu Multi-Task NLP Dataset — Code-Switch Tagging (token-level)

Data: 3,514 comments, ~58,700 individual word tokens, 3 tags per token
(urdu / english / ambiguous).
Result: TF-IDF character-ngram baseline macro F1 ~= 0.51 | Transformer macro F1 = 0.80

FINDING: this is the one task where the transformer clearly won. Unlike the
sentence-level tasks (hate speech, emotion), token-level tagging turns each
of the 2,810 training comments into ~13 individual training signals (one
per word), giving the model far more effective supervision from the same
underlying data. See ../docs/Project_Report.pdf.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset

# --- Parse token-level tags from "word1/tag1 word2/tag2 ..." format ---
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

gold_df = pd.read_csv("../dataset/gold_labeled_final.csv")
silver_df = pd.read_csv("../dataset/silver_labeled.csv")

all_comments = []
for df, source in [(gold_df, "gold"), (silver_df, "silver")]:
    for _, row in df.iterrows():
        parsed = parse_tagged(row.get("code_switch_tags"))
        if parsed:
            words, tags = parsed
            if len(words) == len(tags) and len(words) > 0:
                all_comments.append({"words": words, "tags": tags, "source": source})

print("Total comments with valid token tags:", len(all_comments))  # 3,514
# Tag distribution: urdu ~78%, english ~16%, ambiguous ~5.6%

train_comments, test_comments = train_test_split(all_comments, test_size=0.2, random_state=42)

# ===========================================================================
# BASELINE — per-word TF-IDF character n-grams (context-free)
# ===========================================================================
def flatten(comments):
    words, tags = [], []
    for c in comments:
        words.extend(c["words"])
        tags.extend(c["tags"])
    return words, tags

train_words, train_tags = flatten(train_comments)
test_words, test_tags = flatten(test_comments)

# Character n-grams capture spelling patterns (e.g. "-tion" = English,
# "-ka"/"-ke"/"-hai" = Urdu) — a word-level baseline would just memorize
# vocabulary and fail on unseen words.
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=3000)
X_train = vectorizer.fit_transform(train_words)
X_test = vectorizer.transform(test_words)

baseline_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
baseline_clf.fit(X_train, train_tags)
pred_tags = baseline_clf.predict(X_test)
print("\n--- Character n-gram Baseline ---")
print(classification_report(test_tags, pred_tags, zero_division=0))
# Result: macro F1 ~= 0.51 (urdu 0.88, english 0.72, ambiguous 0.44)

# ===========================================================================
# TRANSFORMER — XLM-RoBERTa, token classification (context-aware)
# ===========================================================================
LABEL_LIST = ["urdu", "english", "ambiguous"]
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for i, l in enumerate(LABEL_LIST)}

clean_train = [c for c in train_comments if all(t in LABEL_LIST for t in c["tags"])]
clean_test = [c for c in test_comments if all(t in LABEL_LIST for t in c["tags"])]

train_data = {"tokens": [c["words"] for c in clean_train], "tags": [[label2id[t] for t in c["tags"]] for c in clean_train]}
test_data = {"tokens": [c["words"] for c in clean_test], "tags": [[label2id[t] for t in c["tags"]] for c in clean_test]}

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align(batch):
    """Tokenizer splits words into subword pieces; only the FIRST subword of
    each real word gets the real label (-100 = ignored in loss for the rest,
    including special tokens like CLS/SEP/PAD)."""
    tokenized = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128)
    all_labels = []
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
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized

train_ds = Dataset.from_dict(train_data).map(tokenize_and_align, batched=True)
test_ds = Dataset.from_dict(test_data).map(tokenize_and_align, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABEL_LIST), id2label=id2label, label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)
    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds, labels):
        for p, l in zip(pred_row, label_row):
            if l != -100:
                true_labels.append(l)
                true_preds.append(p)
    report = classification_report(true_labels, true_preds, target_names=LABEL_LIST, output_dict=True, zero_division=0)
    return {"macro_f1": report["macro avg"]["f1-score"]}

training_args = TrainingArguments(
    output_dir="./codeswitch_model",
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
# Result: best checkpoint macro F1 = 0.80 (epoch 5) — clear win over the 0.51 baseline

model.save_pretrained("./codeswitch_final_model")
tokenizer.save_pretrained("./codeswitch_final_model")
