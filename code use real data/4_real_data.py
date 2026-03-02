import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np

# --- 1. ĐỊNH NGHĨA MÔ HÌNH DEEPFM ---


class DeepFMRankerPro(Model):
    def __init__(self, feature_dims, embedding_dim=8):
        super(DeepFMRankerPro, self).__init__()
        # Embedding cho từng cột dữ liệu Categorical
        self.embeddings = [layers.Embedding(input_dim=dim, output_dim=embedding_dim)
                           for dim in feature_dims]

        # Tuyến DNN (Deep)
        self.dnn = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        # Tuyến FM (Linear)
        self.linear = layers.Dense(1)

    def call(self, inputs):
        cat_inputs, num_inputs = inputs
        # Nhúng (Embedding)
        embedded = [self.embeddings[i](cat_inputs[:, i])
                    for i in range(len(self.embeddings))]
        concat_embed = tf.stack(embedded, axis=1)  # [Batch, Features, Dim]

        # Deep path
        deep_in = tf.concat(
            [layers.Flatten()(concat_embed), num_inputs], axis=1)
        deep_out = self.dnn(deep_in)

        # FM path
        fm_out = self.linear(tf.reduce_sum(concat_embed, axis=1))

        return tf.nn.sigmoid(deep_out + fm_out)

# --- 2. CHUẨN BỊ DỮ LIỆU VÀ SỬA LỖI MAPPING ---


# Đọc bảng tương tác bạn đã nhập tay
df_inter = pd.read_csv('user_interactions.csv')

# Ánh xạ ID thực tế (101, 102...) sang chỉ số mảng (0, 1, 2...)
song_id_map = {id: idx for idx, id in enumerate(df_songs_c['song_id'].values)}
user_id_map = {id: idx for idx, id in enumerate(df_users['user_id'].values)}

# Chuyển đổi ID trong file tương tác
u_indices = df_inter['user_id'].map(user_id_map)
s_indices = df_inter['song_id'].map(song_id_map)

# Loại bỏ các dòng chứa ID không tồn tại trong kho gốc để tránh lỗi Index
valid_mask = u_indices.notna() & s_indices.notna()
u_idx_final = u_indices[valid_mask].astype(int).values
s_idx_final = s_indices[valid_mask].astype(int).values
y_train = df_inter['label'][valid_mask].values

# 3. TRÍCH XUẤT ĐẶC TRƯNG (FEATURES)
# Tạo ma trận đầu vào Categorical [User_ID, User_Genre, User_Device, Song_ID, Song_Genre, Song_Mood]
X_train_cat = np.stack([
    u_idx_final,
    u_cat_features[u_idx_final, 0],
    u_cat_features[u_idx_final, 1],
    s_idx_final,
    s_cat_features[s_idx_final, 0],
    s_cat_features[s_idx_final, 1]
], axis=1)

# Tạo ma trận đầu vào Numerical (Age, BPM, ...)
X_train_num = np.hstack(
    [u_num_features[u_idx_final], s_num_features[s_idx_final]])

# --- 4. KHỞI TẠO VÀ HUẤN LUYỆN ---

# Kích thước dựa trên báo cáo hệ thống của bạn
# cat_dims = [Số User, Số Genre, Số Device, Số Bài hát, Số Genre, Số Mood]
cat_dims = [4, 6, 3, 5, 6, 4]

ranker = DeepFMRankerPro(cat_dims)
ranker.compile(optimizer='adam', loss='binary_crossentropy',
               metrics=['accuracy'])

print("--- Đang huấn luyện DeepFM Ranker ---")
ranker.fit([X_train_cat, X_train_num], y_train,
           epochs=20, batch_size=4, verbose=1)

print("\nFile 4: Hoàn tất! Mô hình đã học được cách xếp hạng từ dữ liệu tay của bạn.")
