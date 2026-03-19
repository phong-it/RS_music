# service_c_content.py

from fastapi import FastAPI
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# ===============================
# LOAD DATA (chỉ load 1 lần)
# ===============================
df_content = pd.read_csv("song_content.csv")
df_context = pd.read_csv("user_context.csv")

# Encode categorical
le_genre = LabelEncoder()
le_mood = LabelEncoder()

df_content["genre_id"] = le_genre.fit_transform(df_content["genre"])
df_content["mood_id"] = le_mood.fit_transform(df_content["mood"])


# ===============================
# TẦNG 1A – CONTENT FEATURE
# ===============================

def extract_content_features(song_id: int):
    song_data = df_content[df_content["song_id"] == song_id]
    if song_data.empty: return None

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
        "mood_id": int(mood_id),
        "genre_id": int(genre_id),
        "content_score": float(content_score)
    }


# ===============================
# TẦNG 1B – CONTEXT FEATURE
# ===============================

def extract_context_features(user_id: int):
    user_data = df_context[df_context["user_id"] == user_id]
    if user_data.empty: return None

    hour = int(user_data["time_of_day"].iloc[0])
    device = user_data["device"].iloc[0]
    movement = user_data["movement_speed"].iloc[0]
    location = user_data["location"].iloc[0]

    if 6 <= hour <= 11: 
        base_energy = 0.7 
    elif 18 <= hour <= 23: 
        base_energy = 0.3 
    else: 
        base_energy = 0.5 

    move_mod = 0.3 if movement == "Running" else (0.1 if movement == "Walking" else 0.0)

    loc_mod = 0.2 if location == "Gym" else (-0.2 if location == "Office" else 0.0)

    dev_mod = 0.1 if device == "Bluetooth Speaker" else 0.0

    target_energy = base_energy + move_mod + loc_mod + dev_mod
    target_energy = max(0.0, min(1.0, target_energy))

    return {
        "time_of_day": hour,
        "device": device,
        "movement_speed": movement,
        "location": location,
        "target_energy": float(target_energy) 
    }


# ===============================
# TẦNG 2 – CASCADE + MIXED FUSION
# ===============================

def fuse_content_context(content, context):
    # Thuật toán so khớp: Tính độ lệch giữa "Năng lượng bài hát" và "Năng lượng người dùng cần"
    energy_diff = abs(content["content_score"] - context["target_energy"])
    
    # Final score là % độ khớp (Độ lệch càng nhỏ -> Điểm càng cao)
    final_score = 1.0 - energy_diff

    return float(final_score)


# ===============================
# API ENDPOINT
# ===============================

@app.get("/analyze/{user_id}/{song_id}")
def analyze(user_id: int, song_id: int):
    content_feature = extract_content_features(song_id)
    context_feature = extract_context_features(user_id)

    if content_feature is None: return {"error": "Song not found"}
    if context_feature is None: return {"error": "User not found"}

    final_score = fuse_content_context(content_feature, context_feature)

    return {
        "song_id": song_id,
        "content_feature": content_feature,
        "context_feature": context_feature,
        "match_score": final_score # Điểm số độ khớp
    }
