"""
Roman Urdu Multi-Task NLP Dataset — Stage 2: Cleaning + Roman Urdu Filtering
Takes the raw scraped comments and filters down to genuine, clean,
Roman-Urdu / code-switched comments.
"""

import re
import pandas as pd

df = pd.read_csv("raw_comments_combined.csv")

# 1. Drop exact duplicates (spam/bot comments repeat often)
df = df.drop_duplicates(subset="comment_text").reset_index(drop=True)

# 2. Remove URL/spam comments
df = df[~df["comment_text"].str.contains(r"http\S+|www\.", regex=True, na=False)]

# 3. Remove emoji/symbol-only comments (require at least 5 real letters)
df["alpha_char_count"] = df["comment_text"].apply(lambda x: len(re.findall(r"[a-zA-Z]", str(x))))
df = df[df["alpha_char_count"] >= 5]

# 4. Word count (used both for filtering and later for the annotation tool)
df["word_count"] = df["comment_text"].apply(lambda x: len(str(x).split()))

# 5. Roman Urdu detector: common Roman Urdu function words, including the
#    informal shorthand spellings people actually use online (k/ky/ny/sy/pe
#    etc, not just the "full" spellings like ke/kya/ne/se/per).
ROMAN_URDU_MARKERS = [
    "hai", "hain", "tha", "thi", "ka", "ki", "ke", "ko", "se", "sy", "mein",
    "aur", "bhi", "nahi", "nhe", "nhi", "kyun", "ky", "kya", "acha", "achi",
    "bura", "wala", "wali", "raha", "rahi", "kar", "krna", "krta", "krti",
    "hoga", "hogi", "hoti", "bhai", "yaar", "per", "pe", "ny", "ne", "k",
    "hy", "tu", "tum", "ap", "aap", "mujhe", "mujy", "unho", "unhon",
    "sab", "jo", "jab", "tak", "bnda", "bndy", "waqt"
]
pattern = r"\b(?:" + "|".join(ROMAN_URDU_MARKERS) + r")\b"
df["has_roman_urdu"] = df["comment_text"].str.lower().str.contains(pattern, regex=True, na=False)

# 6. Final keep-rule: has Roman Urdu markers AND at least 4 words
#    (short "lol yaar"-style reactions carry no real sarcasm/emotion/hate
#    signal even if technically Roman Urdu)
final_df = df[(df["has_roman_urdu"]) & (df["word_count"] >= 4)].reset_index(drop=True)

print("Raw comments:", len(pd.read_csv("raw_comments_combined.csv")))
print("After cleaning + filtering:", len(final_df))  # 3,521

final_df.to_csv("roman_urdu_filtered.csv", index=False)
