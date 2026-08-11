"""
Roman Urdu Multi-Task NLP Dataset — Stage 1: Data Collection
Pulls comments from YouTube videos via the YouTube Data API v3.

Run in Google Colab. Requires a free YouTube Data API key from
Google Cloud Console (console.cloud.google.com -> enable "YouTube Data
API v3" -> Credentials -> Create API key).
"""

!pip install google-api-python-client -q

from googleapiclient.discovery import build
import pandas as pd
import re

API_KEY = "PASTE_YOUR_YOUTUBE_API_KEY_HERE"  # never commit this to GitHub
youtube = build("youtube", "v3", developerKey=API_KEY)

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_comments(video_id, max_comments=600):
    """Fetch top-level comments only. No usernames collected (anonymization
    by design, not stripped out later)."""
    comments = []
    next_page_token = None
    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                pageToken=next_page_token, textFormat="plainText"
            )
            response = request.execute()
        except Exception as e:
            print(f"  Skipping {video_id}: {e}")
            break
        for item in response["items"]:
            top_comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "comment_text": top_comment["textDisplay"],
                "like_count": top_comment["likeCount"],
            })
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments

# Video sources: a deliberate mix of political/news commentary (for hate
# speech + misinformation-adjacent content) and comedy/drama-reaction/vlog
# content (for sarcasm + emotion + natural Roman Urdu density).
VIDEO_URLS = [
    # 11 initial videos (political/news commentary weighted)
    "https://www.youtube.com/watch?v=WJxfQWiP9DQ",
    "https://www.youtube.com/watch?v=BljaIOvFkcM",
    "https://www.youtube.com/watch?v=UiYpRBQ74fs",
    "https://www.youtube.com/watch?v=4k0ZV0UFZSw",
    "https://www.youtube.com/watch?v=HeZ2VBdJeOk",
    "https://www.youtube.com/watch?v=-w7eMfsuRgA",
    "https://www.youtube.com/watch?v=ttL6k7wzGmQ",
    "https://www.youtube.com/watch?v=Wzt1GSye6MU",
    "https://www.youtube.com/watch?v=zXAAAXogJ78",
    "https://www.youtube.com/watch?v=BaMLBWWvNHA",
    "https://www.youtube.com/watch?v=N-0DINjOMlY",
    # 14 additional videos (comedy/drama-reaction/vlog weighted — added
    # after the first batch skewed too heavily pure-English/political)
    "https://www.youtube.com/watch?v=nqbNLe84dpM",
    "https://www.youtube.com/watch?v=XHQrr9fmTkc",
    "https://www.youtube.com/watch?v=gvG35CxYOLg",
    "https://www.youtube.com/watch?v=8ksTrbtyTco",
    "https://www.youtube.com/watch?v=BmpRnfu16YA",
    "https://www.youtube.com/watch?v=FH8OUP2_sN4",
    "https://www.youtube.com/watch?v=VUKai4CZSNw",
    "https://www.youtube.com/watch?v=hhFGhAo4ZfY",
    "https://www.youtube.com/watch?v=EyrNOfyxSBQ",
    "https://www.youtube.com/watch?v=eY0i4nBXqdY",
    "https://www.youtube.com/watch?v=zfjRK9s0pjA",
    "https://www.youtube.com/watch?v=Erif7Vj9NSQ",
    "https://www.youtube.com/watch?v=PKFU1F2QEMQ",
    "https://www.youtube.com/watch?v=IMnXqYDHhZI",
]

all_comments = []
for url in VIDEO_URLS:
    vid = extract_video_id(url)
    if vid:
        print(f"Fetching comments for {vid}...")
        all_comments.extend(get_comments(vid))

df = pd.DataFrame(all_comments)
print(f"Total comments collected: {len(df)}")  # 13,004 across both batches
df.to_csv("raw_comments_combined.csv", index=False)
