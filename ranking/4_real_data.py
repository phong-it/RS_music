import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np


class DeepFMRankerPro(Model):
    def __init__(self, feature_dims, embedding_dim=8):
        super(DeepFMRankerPro, self).__init__()
        # 1. Embedding Layers cho dữ liệu Categorical
        self.embeddings = [layers.Embedding(input_dim=dim, output_dim=embedding_dim)
                           for dim in feature_dims]

        # 2. Tuyến Deep (DNN) - Học tương tác phức tạp
        self.dnn = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        # 3. Tuyến FM (Linear) - Học tương tác cơ bản
        self.linear = layers.Dense(1)

    def call(self, inputs):
        cat_inputs, num_inputs = inputs

        # Nhúng các ID thành vector
        embedded = [self.embeddings[i](cat_inputs[:, i])
                    for i in range(len(self.embeddings))]
        concat_embed = tf.stack(embedded, axis=1)  # [Batch, Features, Dim]

        # Tuyến Deep: Kết hợp Embedding phẳng + Dữ liệu số (Age, BPM...)
        deep_in = tf.concat(
            [layers.Flatten()(concat_embed), num_inputs], axis=1)
        deep_out = self.dnn(deep_in)

        # Tuyến FM: Cộng gộp các vector tương tác
        fm_out = self.linear(tf.reduce_sum(concat_embed, axis=1))

        return tf.nn.sigmoid(deep_out + fm_out)


# --- BƯỚC HUẤN LUYỆN VỚI DỮ LIỆU THẬT ---
# Định nghĩa số lượng nhãn dựa trên dữ liệu từ File 1
cat_dims = [NUM_USERS, NUM_GENRES, 3, NUM_SONGS, NUM_GENRES, 4]
ranker = DeepFMRankerPro(cat_dims)
ranker.compile(optimizer='adam', loss='binary_crossentropy',
               metrics=['accuracy'])

# 1. Đọc bảng tương tác (Đáp án)
df_inter = pd.read_csv('user_interactions.csv')

# 2. Chuẩn bị dữ liệu Train (Ánh xạ ID sang Feature Matrix từ File 1)
u_ids_train = df_inter['user_id'].values
s_ids_train = df_inter['song_id'].values

X_train_cat = np.stack([
    u_ids_train, u_cat_features[u_ids_train,
                                0], u_cat_features[u_ids_train, 1],
    s_ids_train, s_cat_features[s_ids_train, 0], s_cat_features[s_ids_train, 1]
], axis=1)

X_train_num = np.hstack(
    [u_num_features[u_ids_train], s_num_features[s_ids_train]])
y_train = df_inter['label'].values

# 3. Thực hiện huấn luyện
print("Đang huấn luyện DeepFM...")
ranker.fit([X_train_cat, X_train_num], y_train,
           epochs=10, batch_size=32, verbose=1)
print("File 4: Hoàn tất huấn luyện mô hình Ranking.")
