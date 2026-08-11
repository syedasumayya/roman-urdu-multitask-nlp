# Roman Urdu Multi-Task NLP

A benchmark dataset and set of experiments for **Roman Urdu / English
code-switched text** — the informal mixed-language writing millions of
Pakistanis use daily on YouTube, WhatsApp, and social media, and which
remains significantly under-resourced in NLP research.

This project studies five tasks on one shared dataset — **sarcasm
detection, hate speech classification, misinformation flagging, emotion
recognition, and code-switch tagging** — and asks a deeper question than
"can we build a classifier": **when and why do large pretrained
transformers fail to beat simple baselines in a low-resource, code-switched
language, and does training multiple tasks together help or hurt?**

---

## Key Finding

Across every experiment here, a consistent pattern holds: **model
sophistication is not a substitute for training data volume.** A simple
TF-IDF + Logistic Regression baseline matched or beat a fine-tuned
XLM-RoBERTa transformer in 3 of 5 tasks. The one clear transformer win
(code-switch tagging) succeeded specifically because its token-level
framing provides far more effective training signal per comment (~13
labels per comment) than sentence-level tasks (1 label per comment).
Naive multi-task learning caused **negative transfer** — one high-signal
task dominated the shared model at the expense of the others — and
loss-reweighting revealed a genuine trade-off rather than a clean fix.

Full results: [`results/full_results_summary.md`](results/full_results_summary.md)

---

## Dataset

| | |
|---|---|
| Raw comments collected | 13,004 (25 YouTube videos) |
| After cleaning + Roman Urdu filtering | 3,521 |
| **Gold** (human-labeled, all 5 tasks) | 600 |
| **Silver** (LLM-labeled, 4 of 5 tasks — sarcasm excluded, see below) | 2,918 |

**Sources:** a deliberate mix of political/news commentary videos (for
hate speech and misinformation-adjacent content) and comedy/drama-reaction/
vlog content (for sarcasm, emotion, and natural Roman Urdu density).

**Ethics note:** only comment text and like counts were collected — no
usernames, profile photos, or other identifying information, by design
from the start.

### Why sarcasm is gold-only

Before using an LLM to label sarcasm at scale, we validated LLM judgments
against the 600 human gold labels. Three configurations were tested:

| Configuration | Agreement with human labels |
|---|---|
| Claude Haiku (basic prompt) | 69.9% |
| Claude Haiku (few-shot calibrated) | 75.5% |
| Claude Sonnet (stronger model) | 79.1% |
| Majority-class baseline (always "not sarcastic") | 80.0% |

**All three LLM configurations failed to beat trivially guessing "not
sarcastic."** Rather than silently ship unreliable labels, sarcasm is
modeled using only the 600 human-labeled gold comments.

---

## Results

### Single-task

| Task | Baseline (TF-IDF) | Transformer | Winner |
|---|---|---|---|
| Hate speech | 0.57 | 0.56 | Tie |
| Emotion (multi-label) | 0.46 | 0.40 | Baseline |
| Code-switch tagging (token-level) | ~0.51 | **0.80** | **Transformer** |
| Sarcasm (gold-only, 600 examples) | **0.59** | 0.44 (collapsed to baseline) | Baseline |
| Misinformation | — | — | Not viable (only 5 positive examples) |

*(All scores are macro F1.)*

### Multi-task (shared encoder, 3 tasks jointly)

| Task | Single-task best | Multi-task (equal weight) | Multi-task (reweighted) |
|---|---|---|---|
| Hate speech | 0.57 | 0.55 | 0.53 |
| Emotion | 0.46 | 0.28 | 0.31 |
| Code-switch tagging | 0.80 | 0.78 | 0.66 |
| Average | — | 0.535 | 0.501 |

Reweighting (up-weighting emotion, down-weighting tagging) partially
recovered emotion but degraded tagging more than proportionally — a
genuine trade-off, not a fix. Full discussion in the results summary.

---

## Repo Structure

```
RomanUrdu-MultiTask-NLP/
├── README.md
├── dataset/
│   ├── gold_labeled_final.csv      <- 600 human-labeled comments, all 5 tasks
│   └── silver_labeled.csv          <- 2,918 LLM-labeled comments, 4 tasks
├── annotation_tool/
│   └── roman_urdu_annotation_tool.html   <- browser-based tool used for gold labeling
├── code/
│   ├── 1_data_collection.py
│   ├── 2_data_cleaning.py
│   ├── 3_gold_silver_split_and_labeling.py
│   ├── 4_train_hate_speech.py
│   ├── 5_train_emotion.py
│   ├── 6_train_codeswitch.py
│   ├── 7_train_sarcasm.py
│   └── 8_train_multitask.py
├── results/
│   └── full_results_summary.md
├── models/
│   └── README.md                   <- how to host/load the trained models (too large for git)
└── docs/
    └── Project_Report.pdf          <- full write-up
```

## How to Reproduce

All training was done in Google Colab with a T4 GPU.

1. **Data collection** (`code/1_data_collection.py`) — needs a free YouTube
   Data API v3 key from Google Cloud Console.
2. **Cleaning** (`code/2_data_cleaning.py`).
3. **Gold/silver split + silver labeling** (`code/3_gold_silver_split_and_labeling.py`)
   — needs an Anthropic API key. Gold labeling itself is done manually
   using the annotation tool in `annotation_tool/`.
4. **Single-task models** (`code/4_train_hate_speech.py` through
   `code/7_train_sarcasm.py`) — each script trains both a TF-IDF baseline
   and a transformer, and prints the comparison.
5. **Multi-task model** (`code/8_train_multitask.py`) — trains both the
   equal-weight and reweighted configurations.

Since the dataset (`dataset/`) is already included in this repo, you can
skip straight to step 4 or 5 if you just want to reproduce the modeling
results without re-collecting or re-labeling data.

## Annotation Tool

`annotation_tool/roman_urdu_annotation_tool.html` is a self-contained
browser tool built for this project — no installation needed, just open
it in a browser. It shows one comment at a time with:
- Clickable word-by-word tagging for code-switch labels
- Single-select buttons for sarcasm, hate speech, and misinformation
- Multi-select for emotion
- Progress tracking and CSV export (with a copy-paste fallback for
  sandboxed environments that block file downloads)

## Limitations

- **Misinformation** was not modeled — the gold set had only 5 comments
  that were genuine factual claims (out of 600), far too few to train or
  meaningfully evaluate.
- **Sarcasm** modeling is limited to 600 gold examples; a larger
  human-labeled set would likely improve results further.
- Silver labels (LLM-generated) were spot-validated against gold but are
  not independently human-verified at full scale.
- Multi-task experiments used only 2 loss-weighting configurations;
  gradient-based task-balancing methods were not attempted.

## Future Work

- Gradient-based multi-task balancing (e.g. GradNorm, PCGrad) to properly
  address the negative transfer observed here
- Collecting more gold-labeled sarcasm examples specifically
- A dedicated misinformation-focused data collection pass (targeting
  claim-heavy content, e.g. news comment sections specifically)
- Testing Urdu-specific or South-Asian-language-specific pretrained models
  as an alternative to general multilingual XLM-RoBERTa
