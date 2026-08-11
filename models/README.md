# Trained Models

Model weights are **not included directly in this Git repo** — each fine-tuned
XLM-RoBERTa model is ~1.1GB, well over GitHub's file size limits even with Git LFS
being impractical at this scale for free accounts.

## What you should have locally (downloaded during training)

- `hate_speech_final_model.zip`
- `emotion_final_model.zip`
- `codeswitch_final_model.zip`
- `sarcasm_final_model.zip`
- `multitask_model_v1_final.pt` + `multitask_tokenizer.zip` (equal-weight run)
- `multitask_model_v2_final.pt` + `multitask_tokenizer_v2.zip` (reweighted run)

## Recommended: host on Hugging Face Hub instead of GitHub

This is the standard practice for sharing trained NLP models, and it's free:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Install the CLI: `pip install huggingface_hub`
3. Log in: `huggingface-cli login`
4. For each model folder (after unzipping):
   ```bash
   huggingface-cli upload your-username/roman-urdu-hate-speech ./hate_speech_final_model
   ```
5. Repeat for each of the 6 models, using descriptive repo names
   (e.g. `roman-urdu-hate-speech`, `roman-urdu-emotion`, `roman-urdu-codeswitch`,
   `roman-urdu-sarcasm`, `roman-urdu-multitask-v1`, `roman-urdu-multitask-v2`)
6. Add the resulting Hugging Face links to this README once uploaded, and reference
   them from the main project README

## Loading a model later (example: hate speech)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("your-username/roman-urdu-hate-speech")
tokenizer = AutoTokenizer.from_pretrained("your-username/roman-urdu-hate-speech")
```

## Loading the multi-task model

The multi-task model uses a custom architecture (see `code/8_train_multitask.py`), so
it can't be loaded with `AutoModel` directly — you'll need the `MultiTaskModel` class
definition from that script alongside the saved `.pt` state dict:

```python
from code.train_multitask import MultiTaskModel  # adjust import path as needed
import torch

model = MultiTaskModel("xlm-roberta-base")
model.load_state_dict(torch.load("multitask_model_v1_final.pt"))
```
