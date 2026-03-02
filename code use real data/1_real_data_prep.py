import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. ĐỌC DỮ LIỆU THẬT TỪ CÁC BÊN ---
# Giả sử bạn có các tệp CSV xuất ra từ hệ thống của A, B, C
try:
    # Gồm: user_id, fav_genre, age
    df_user_a = pd.read_csv('user_static_A.csv')
    # Gồm: user_id, skip_rate, device_type
    df_user_b = pd.read_csv('user_behavior_B.csv')
    # Gồm: song_id, bpm, mood, genre
    df_songs_c = pd.read_csv('song_content_C.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp dữ liệu. Hãy đảm bảo các tệp CSV đã sẵn sàng.")
    # (Giữ code random cũ làm fallback nếu cần)

# --- 2. HỢP NHẤT DỮ LIỆU (JOIN) ---
# Kết hợp dữ liệu tĩnh (A) và hành vi (B) của người dùng dựa trên user_id
df_users = pd.merge(df_user_a, df_user_b, on='user_id')

# --- 3. XỬ LÝ DỮ LIỆU PHÂN LOẠI (ENCODING) ---
# Tầng AI cần số nguyên (0, 1, 2...) thay vì chữ ('Pop', 'Rock'...)
genre_encoder = LabelEncoder()
mood_encoder = LabelEncoder()
device_encoder = LabelEncoder()

# Chuyển đổi tên thể loại sang ID
df_users['genre_id'] = genre_encoder.fit_transform(df_users['fav_genre'])
df_users['device_id'] = device_encoder.fit_transform(df_users['device_type'])

df_songs_c['genre_id'] = genre_encoder.transform(
    df_songs_c['genre'])  # Dùng chung encoder với User
df_songs_c['mood_id'] = mood_encoder.fit_transform(df_songs_c['mood'])

# Xử lý giá trị thiếu (Handling NaNs)
# Điền các ô trống bằng giá trị trung bình
df_users = df_users.fillna(df_users.mean(numeric_only=True))

# Cập nhật cấu hình dựa trên dữ liệu thật
NUM_USERS = len(df_users)
NUM_SONGS = len(df_songs_c)
NUM_GENRES = len(genre_encoder.classes_)

# --- 4. CHUẨN HÓA DỮ LIỆU SỐ (SCALING) ---
scaler_u = StandardScaler()
scaler_s = StandardScaler()

# Chuẩn hóa Age và Skip_Rate (User)
u_num_features = scaler_u.fit_transform(df_users[['age', 'skip_rate']].values)

# Chuẩn hóa BPM (Song)
s_num_features = scaler_s.fit_transform(df_songs_c[['bpm']].values)

# --- 5. TỔNG HỢP FEATURE VECTOR ---
# Dữ liệu Categorical (ID)
u_cat_features = df_users[['genre_id', 'device_id']].values.astype('int32')
s_cat_features = df_songs_c[['genre_id', 'mood_id']].values.astype('int32')

print(
    f"File 1: Đã tải {NUM_USERS} người dùng và {NUM_SONGS} bài hát từ dữ liệu thật.")
print(f"Danh sách thể loại: {list(genre_encoder.classes_)}")
