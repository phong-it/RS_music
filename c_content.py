# service_c_content.py

from fastapi import FastAPI
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# ===============================
# LOAD DATA (chỉ load 1 lần lúc bật server)
# ===============================
df_content = pd.read_csv("song_content.csv")
df_context = pd.read_csv("user_context.csv")

# Encode categorical
le_genre = LabelEncoder()
le_mood = LabelEncoder()

# ĐÃ SỬA: Thay df thành df_content
df_content["genre_id"] = le_genre.fit_transform(df_content["genre"])
df_content["mood_id"] = le_mood.fit_transform(df_content["mood"])


# ===============================
# TẦNG 1A – CONTENT FEATURE
# ===============================

def extract_content_features(song_id: int):

    song_data = df_content[df_content["song_id"] == song_id]

    if song_data.empty:
        return None

    bpm = song_data["bpm"].mean()
    mood_id = song_data["mood_id"].iloc[0]
    genre_id = song_data["genre_id"].iloc[0]

    bpm_norm = bpm / 180.0

    max_mood = df_content["mood_id"].max()
    max_mood = max_mood if max_mood > 0 else 1
    
    max_genre = df_content["genre_id"].max()
    max_genre = max_genre if max_genre > 0 else 1

    content_score = (
        0.5 * bpm_norm +
        0.3 * (mood_id / max_mood) +
        0.2 * (genre_id / max_genre)
    )

    return {
        "bpm": float(bpm),
        "bpm_norm": float(bpm_norm),
        "mood_id": int(mood_id),
        "genre_id": int(genre_id),
        "content_score": float(content_score)
    }


# ===============================
# TẦNG 1B – CONTEXT FEATURE
# ===============================

def extract_context_features(user_id: int):

    user_data = df_context[df_context["user_id"] == user_id]

    if user_data.empty:
        return None

    hour = user_data["hour"].mean()

    if 6 <= hour <= 11:
        time_weight = 1.0
    elif 18 <= hour <= 23:
        time_weight = 0.7
    else:
        time_weight = 0.5

    avg_skip = user_data["skip_rate"].mean()

    return {
        "time_weight": time_weight,
        "avg_skip_rate": float(avg_skip)
    }


# ===============================
# TẦNG 2 – CASCADE + MIXED FUSION
# ===============================

def fuse_content_context(content, context):

    # Nếu user skip nhiều → giảm điểm
    skip_penalty = 1 - context["avg_skip_rate"]

    final_score = (
        0.6 * content["content_score"] +
        0.2 * context["time_weight"] +
        0.2 * skip_penalty
    )

    return float(final_score)


# ===============================
# API ENDPOINT
# ===============================

@app.get("/analyze/{user_id}/{song_id}")
def analyze(user_id: int, song_id: int):

    content_feature = extract_content_features(song_id)
    context_feature = extract_context_features(user_id)

    if content_feature is None:
        return {"error": "Song not found"}

    if context_feature is None:
        return {"error": "User not found"}

    final_score = fuse_content_context(content_feature, context_feature)

    return {
        "song_id": song_id,
        "content_feature": content_feature,
        "context_feature": context_feature,
        "final_contextual_score": final_score
    }
