# !pip install faiss-cpu numpy pandas tensorflow scikit-learn

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cấu hình hệ thống
NUM_USERS = 1000
NUM_SONGS = 5000
NUM_GENRES = 6
genres = ['Pop', 'Rock', 'Indie', 'Lofi', 'Jazz', 'EDM']

# --- 1. NGƯỜI A & B: DỮ LIỆU NGƯỜI DÙNG ---
u_fav_genre = np.random.randint(
    0, NUM_GENRES, size=(NUM_USERS, 1))  # Categorical
u_device = np.random.randint(0, 3, size=(
    NUM_USERS, 1))             # Categorical
u_age = np.random.randint(15, 75, size=(NUM_USERS, 1))             # Numerical
u_skip = np.random.rand(NUM_USERS, 1).astype('float32')            # Numerical

# --- 2. NGƯỜI C: DỮ LIỆU BÀI HÁT ---
s_genre = np.random.randint(
    0, NUM_GENRES, size=(NUM_SONGS, 1))    # Categorical
s_mood = np.random.randint(0, 4, size=(NUM_SONGS, 1)
                           )              # Categorical
s_bpm = np.random.randint(50, 220, size=(NUM_SONGS, 1))            # Numerical

# --- 3. CHUẨN HÓA CHUYÊN NGHIỆP (Fix lỗi chia cứng) ---
scaler_u = StandardScaler()
scaler_s = StandardScaler()

# Tách biệt đầu vào: Categorical (ID) và Numerical (Đã chuẩn hóa)
u_cat_features = np.hstack([u_fav_genre, u_device])
u_num_features = scaler_u.fit_transform(np.hstack([u_age, u_skip]))

s_cat_features = np.hstack([s_genre, s_mood])
s_num_features = scaler_s.fit_transform(s_bpm)

print("File 1: Đã chuẩn hóa động và tách biệt dữ liệu Categorical/Numerical.")
