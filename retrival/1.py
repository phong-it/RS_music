# !pip install faiss-cpu numpy pandas tensorflow

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
import faiss

# Cấu hình hệ thống
NUM_USERS = 1000
NUM_SONGS = 5000
EMBEDDING_DIM = 32

# --- NGƯỜI A: DỮ LIỆU TƯỜNG MINH (EXPLICIT DATA) ---
genres = ['Pop', 'Rock', 'Indie', 'Lofi', 'Jazz', 'EDM']


def get_user_static_data():
    # Giả lập: Mỗi user có 1 thể loại yêu thích nhất (chuyển sang số)
    user_fav_genre = np.random.randint(0, len(genres), size=NUM_USERS)
    # Tuổi của user (từ 15 đến 60)
    user_age = np.random.randint(
        15, 60, size=NUM_USERS) / 60.0  # Chuẩn hóa về [0,1]
    return user_fav_genre, user_age

# --- NGƯỜI B: DỮ LIỆU HÀNH VI (BEHAVIORAL DATA) ---


def get_user_behavior_data():
    # Tỷ lệ skip nhạc trung bình của user (0 đến 1)
    skip_rate = np.random.rand(NUM_USERS)
    # Thiết bị hay dùng (0: Mobile, 1: Desktop, 2: Web)
    device_type = np.random.randint(0, 3, size=NUM_USERS)
    return skip_rate, device_type

# --- NGƯỜI C: DỮ LIỆU NỘI DUNG & NGỮ CẢNH (CONTENT & CONTEXT) ---


def get_song_content_data():
    # BPM chuẩn hóa (nhịp tim bài hát)
    bpm = np.random.randint(60, 180, size=NUM_SONGS) / 180.0
    # Mood bài hát (0: Buồn, 1: Vui, 2: Chill, 3: Sung sức)
    song_mood = np.random.randint(0, 4, size=NUM_SONGS)
    # Thể loại bài hát
    song_genre = np.random.randint(0, len(genres), size=NUM_SONGS)
    return bpm, song_mood, song_genre


# --- TỔNG HỢP DATA ĐẦU VÀO CHO NGƯỜI D (PHONG) ---
u_fav, u_age = get_user_static_data()
u_skip, u_dev = get_user_behavior_data()
s_bpm, s_mood, s_genre = get_song_content_data()

# Tạo Feature Vector cho User (Ghép dữ liệu A và B)
# Vector: [Fav_Genre_ID, Age, Skip_Rate, Device_ID]
user_features = np.stack([u_fav, u_age, u_skip, u_dev],
                         axis=1).astype('float32')

# Tạo Feature Vector cho Song (Dữ liệu từ C)
# Vector: [BPM, Mood_ID, Genre_ID]
song_features = np.stack([s_bpm, s_mood, s_genre], axis=1).astype('float32')

print(f"User Feature Sample (A+B): {user_features[0]}")
print(f"Song Feature Sample (C): {song_features[0]}")
