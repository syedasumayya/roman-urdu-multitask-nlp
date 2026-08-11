# Full Results Summary

## Dataset
- 13,004 raw YouTube comments collected (25 videos: political/news commentary + comedy/drama-reaction/vlog content)
- 3,521 cleaned, genuine Roman Urdu / code-switched comments after filtering
- **600 gold** comments — fully hand-labeled by a human annotator across all 5 tasks
- **2,918 silver** comments — labeled via Claude Haiku (few-shot calibrated prompt), validated against gold

## Sarcasm LLM Validation Experiment
Before committing to LLM-assisted labeling for sarcasm, three configurations were tested
against the 600 human gold labels:

| Configuration | Agreement with human labels |
|---|---|
| Claude Haiku (basic prompt) | 69.9% |
| Claude Haiku (few-shot calibrated) | 75.5% |
| Claude Sonnet (more capable model) | 79.1% |
| **Majority-class baseline (always "no")** | **80.0%** |

**All three LLM configurations failed to beat the trivial majority baseline.** Decision:
sarcasm is excluded from the LLM-labeled silver set entirely and modeled using only the
600 gold, human-labeled comments.

## Single-Task Modeling Results

| Task | Data size | Baseline (TF-IDF) | Transformer (XLM-RoBERTa) | Winner |
|---|---|---|---|---|
| Hate speech (3-class) | 3,514 | 0.57 macro F1 | 0.56 macro F1 | Tie (baseline slightly ahead) |
| Emotion (multi-label, 6 classes) | 3,514 | 0.46 macro F1 | 0.40 macro F1 | Baseline |
| Code-switch tagging (token-level, 3-class) | 3,514 comments / ~58,700 tokens | ~0.51 macro F1 | **0.80 macro F1** | **Transformer (clear win)** |
| Sarcasm (binary, gold-only) | 600 | **0.59 macro F1** | 0.44 macro F1 (collapsed to majority baseline) | Baseline |
| Misinformation | 600 (only 5 positive examples) | — | — | Not viable — documented as data limitation |

### Interpretation
Transformer performance correlates with **effective training signal per example**, not
just raw comment count. Code-switch tagging turns every comment into ~13 word-level
training signals; the sentence-level tasks (hate speech, emotion, sarcasm) get exactly
one signal per comment. With ~2,800-3,500 comments, that's enough effective signal for
code-switch tagging to benefit from a large pretrained model, but not enough for the
sentence-level tasks — where a much simpler TF-IDF baseline matched or beat the
transformer every time.

## Multi-Task Modeling Results

One shared XLM-RoBERTa encoder + 3 task-specific heads (hate speech, emotion,
code-switch tagging), trained jointly on 3,513 examples with all 3 labels present.

| Task | Single-task best | Multi-task (equal weight) | Multi-task (reweighted) |
|---|---|---|---|
| Hate speech | 0.57 | 0.55 | 0.53 |
| Emotion | 0.46 | 0.28 | 0.31 |
| Code-switch tagging | 0.80 | 0.78 | 0.66 |
| **Average** | — | **0.535** | 0.501 |

### Interpretation
**Equal-weight multi-task training caused negative transfer.** Code-switch tagging's much
larger effective sample size let its gradients dominate the shared encoder, degrading the
other two tasks — emotion lost 18 F1 points relative to its single-task score.

**Reweighting the loss (down-weighting tagging, up-weighting emotion) did not fix this —
it revealed a genuine trade-off.** Emotion partially recovered (0.28 → 0.31), but
code-switch tagging degraded more than proportionally (0.78 → 0.66), producing a *lower*
overall average than the naive equal-weighted run. This suggests loss reweighting alone
is insufficient here; more sophisticated multi-task balancing techniques (e.g.
gradient-based methods) or fundamentally more training data per task would likely be
needed to make joint training pay off in this setting.

## Overall Project Finding

Across every experiment in this project, a consistent theme emerges: **in this
low-resource, code-switched setting, model sophistication is not a substitute for
training data volume.** Simple baselines matched or beat fine-tuned transformers in 3 of
5 single-task experiments, and naive multi-task learning produced negative transfer
rather than the hoped-for shared benefit. The one clear transformer win (code-switch
tagging) succeeded specifically because its token-level framing provided far more
effective training signal per comment than any other task.
