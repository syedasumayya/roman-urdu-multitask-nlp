"""
Roman Urdu Multi-Task NLP Dataset — Stage 3: Gold/Silver Split + Silver Labeling

Splits the cleaned 3,521 comments into:
  - 600 "gold" comments, hand-labeled by a human annotator (see
    ../annotation_tool/roman_urdu_annotation_tool.html)
  - ~2,921 "silver" comments, labeled by an LLM (Claude Haiku) with a
    few-shot-calibrated prompt

IMPORTANT — the sarcasm label validation:
  Three different LLM configurations were tested against the 600 human
  gold labels for the sarcasm task specifically:
    - Claude Haiku (basic prompt):            69.9% agreement
    - Claude Haiku (few-shot calibrated):      75.5% agreement
    - Claude Sonnet (more capable model):      79.1% agreement
    - Majority-class baseline (always "no"):   80.0%
  All three LLM configurations FAILED to beat the trivial majority
  baseline. Conclusion: sarcasm is NOT included in the silver-labeled
  set. Sarcasm modeling uses ONLY the 600 gold, human-labeled comments.
  See ../docs/Project_Report.pdf for the full experiment writeup.
"""

import pandas as pd
import numpy as np
import json, re, time
import anthropic

# ---------------------------------------------------------------------------
# STEP 1 — Split into gold (600, manual) and silver (rest, LLM-assisted)
# ---------------------------------------------------------------------------
final_df = pd.read_csv("roman_urdu_filtered.csv")
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

GOLD_SIZE = 600
gold_df = final_df.iloc[:GOLD_SIZE].copy()
silver_df = final_df.iloc[GOLD_SIZE:].copy()

gold_df.to_csv("gold_unlabeled.csv", index=False)
silver_df.to_csv("silver_unlabeled.csv", index=False)
print("Gold set (manual annotation):", len(gold_df))     # 600
print("Silver set (LLM-assisted):", len(silver_df))       # 2,921

# ---------------------------------------------------------------------------
# STEP 2 — Manual gold labeling happens here, using the annotation tool
# (../annotation_tool/roman_urdu_annotation_tool.html). This is done outside
# Python — the tool exports gold_labeled_final.csv, which is what feeds into
# every downstream step of this project.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# STEP 3 — LLM-assisted silver labeling (Claude Haiku)
# ---------------------------------------------------------------------------
API_KEY = "PASTE_YOUR_ANTHROPIC_KEY_HERE"  # never commit this to GitHub
client = anthropic.Anthropic(api_key=API_KEY)

# This is the FINAL, few-shot-calibrated prompt (v2). The original version
# lacked the SARCASTIC/NOT-sarcastic examples below and scored only 69.9%
# agreement on sarcasm specifically; adding real examples pulled from the
# gold set raised that to 75.5% — still short of the 80% baseline, which is
# why sarcasm was ultimately excluded from the silver set (see docstring).
LABEL_PROMPT = """You are annotating a Roman Urdu / English code-switched YouTube comment for a research dataset.

For SARCASM specifically, here are real examples to calibrate your judgment:

SARCASTIC examples (mocking tone, backhanded compliments, ironic exaggeration, crude dismissiveness):
- "1:45 hud 40 plus mardo and fir bolo 2 police wala injured huva hai toba karo g" (mock-scolding tone)
- "My ny unsubscribe kr lia zyada maza araha try kro sb" (ironic — claims to have left, then sarcastically recommends it)
- "Bhaiya ka tatti video" (crude, dismissive mockery, not sincere commentary)

NOT sarcastic examples (sincere, genuine tone even if casual/emoji-heavy):
- "Mashallah aapi bahut accha hai vlog aap ka" (sincere compliment)
- "Hyee Sachi kanwal is the best vloger mujhy ap ki collection bht achi lgi" (genuine fan praise)
- "Fatima api ma ape ki sab sa bade fan kho haya Noor" (sincere fan statement)

Now analyze the comment below and respond with ONLY a JSON object, no other text, no markdown formatting.

Comment: "{comment}"

Words in order: {words}

Provide labels for these 5 tasks:

1. sarcasm: "yes" or "no" — use the calibration examples above. Genuine enthusiasm/praise, even if emoji-heavy or casual, is NOT sarcasm.
2. hate_speech: "none", "offensive", or "hate" (hate = attacks a person/group based on identity; offensive = rude/insulting but not identity-based; none = neither)
3. misinformation: "na" (not making a factual claim), "factual" (makes a claim that is true/accurate), or "misinfo" (makes a claim that is false/misleading)
4. emotion: array of any that apply from ["joy","anger","sadness","fear","surprise","none"] (can be multiple, or ["none"] if neutral)
5. code_switch_tags: for EACH word listed above, in the exact same order, tag it "urdu", "english", or "ambiguous". Return as a single string like "word1/tag1 word2/tag2 ..." — must have exactly {word_count} tagged words, matching the word list exactly.

Respond with ONLY this JSON structure:
{{"sarcasm": "...", "hate_speech": "...", "misinformation": "...", "emotion": [...], "code_switch_tags": "..."}}"""

def label_comment(comment_text):
    words = comment_text.split()
    prompt = LABEL_PROMPT.format(
        comment=comment_text.replace('"', "'"),
        words=", ".join(words),
        word_count=len(words)
    )
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000, temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        text = re.sub(r"^```json\s*|\s*```$", "", response.content[0].text.strip())
        return json.loads(text)
    except Exception as e:
        print(f"  Error labeling comment: {e}")
        return None

results = []
for i, row in silver_df.iterrows():
    labels = label_comment(row["comment_text"])
    if labels:
        results.append({
            "video_id": row.get("video_id", ""), "comment_text": row["comment_text"],
            "sarcasm": labels.get("sarcasm", ""),           # NOTE: excluded from final use, see docstring
            "hate_speech": labels.get("hate_speech", ""),
            "misinformation": labels.get("misinformation", ""),
            "emotion": "|".join(labels.get("emotion", [])),
            "code_switch_tags": labels.get("code_switch_tags", ""),
        })
    else:
        results.append({"video_id": row.get("video_id", ""), "comment_text": row["comment_text"],
                         "sarcasm": "", "hate_speech": "", "misinformation": "", "emotion": "", "code_switch_tags": ""})
    if (i + 1) % 50 == 0:
        pd.DataFrame(results).to_csv("silver_labeled_checkpoint.csv", index=False)
        print(f"Checkpoint: {i+1}/{len(silver_df)} labeled")
    time.sleep(0.1)

pd.DataFrame(results).to_csv("silver_labeled.csv", index=False)
print("Done. Silver labeling complete:", len(results))  # 2,918 successful (4 JSON-parse failures, negligible)
