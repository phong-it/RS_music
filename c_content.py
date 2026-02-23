# service_c_content.py

from fastapi import FastAPI
from feature_store import save_song_feature, save_user_feature
import random
from datetime import datetime

app = FastAPI()

# -------------------------------
# TẦNG 1A – CONTENT FEATURE
# -------------------------------

def extract_content_features(song_id: str):

    bpm = random.randint(60, 160)
    energy = random.uniform(0.3, 1.0)

    mood = "energetic" if bpm > 110 else "calm"

    lyrics_topic_score = random.uniform(0.5, 1.0)

    content_score = (
        0.4 * (bpm / 160) +
        0.4 * energy +
        0.2 * lyrics_topic_score
    )

    return {
        "bpm": bpm,
        "energy": energy,
        "mood": mood,
        "lyrics_score": lyrics_topic_score,
        "content_score": content_score
    }


# -------------------------------
# TẦNG 1B – CONTEXT FEATURE
# -------------------------------

def extract_context_features(user_id: str):

    hour = datetime.now().hour

    if 6 <= hour <= 11:
        time_context = "morning"
        preferred_mood = "energetic"
    elif 18 <= hour <= 23:
        time_context = "evening"
        preferred_mood = "calm"
    else:
        time_context = "neutral"
        preferred_mood = "neutral"

    device = random.choice(["headphones", "speaker"])
    moving = random.choice([True, False])

    return {
        "time_context": time_context,
        "preferred_mood": preferred_mood,
        "device": device,
        "moving": moving
    }


# -------------------------------
# TẦNG 2 – MIXED HYBRID FUSION
# -------------------------------

def fuse_content_context(content, context):

    context_match = 0.0

    if context["preferred_mood"] == content["mood"]:
        context_match += 0.6

    if context["moving"] and content["bpm"] > 120:
        context_match += 0.4

    final_score = (
        0.6 * content["content_score"] +
        0.4 * context_match
    )

    return final_score


# -------------------------------
# API ENDPOINT
# -------------------------------

@app.post("/analyze/{user_id}/{song_id}")
def analyze(user_id: str, song_id: str):

    # Cascade Step 1
    content_feature = extract_content_features(song_id)
    context_feature = extract_context_features(user_id)

    # Mixed Step 2
    final_score = fuse_content_context(content_feature, context_feature)

    # Lưu song feature
    save_song_feature(song_id, {
        **content_feature,
        "context_adjusted_score": final_score
    })

    # Lưu user context
    save_user_feature(user_id, context_feature)

    return {
        "song_id": song_id,
        "content_feature": content_feature,
        "context_feature": context_feature,
        "final_contextual_score": final_score
    }