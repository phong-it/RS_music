import tensorflow as tf
from tensorflow.keras import layers, Model


class DeepFMRankerPro(Model):
    def __init__(self, feature_dims, embedding_dim=8):
        super(DeepFMRankerPro, self).__init__()
        # 1. Embedding Layers cho dữ liệu phân loại (Categorical)
        self.embeddings = [layers.Embedding(input_dim=dim, output_dim=embedding_dim)
                           for dim in feature_dims]

        # 2. Tuyến Deep (DNN) - Học tương tác bậc cao
        self.dnn = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        # 3. Tuyến FM (Học tương tác bậc 1 và 2)
        self.linear = layers.Dense(1)  # Order-1: Tác động riêng lẻ

    def call(self, inputs):
        cat_inputs, num_inputs = inputs  # Nhận cả ID và số đã scale

        # Nhúng các ID thành vector
        embedded = [self.embeddings[i](cat_inputs[:, i])
                    for i in range(len(self.embeddings))]
        # [Batch, Num_Features, Embed_Dim]
        concat_embed = tf.stack(embedded, axis=1)

        # Tuyến Deep: Nối Embedding + Dữ liệu số
        deep_in = tf.concat(
            [layers.Flatten()(concat_embed), num_inputs], axis=1)
        deep_out = self.dnn(deep_in)

        # Tuyến FM (Đơn giản hóa bậc 2): Cộng gộp các vector tương tác
        fm_out = self.linear(tf.reduce_sum(concat_embed, axis=1))

        # Đầu ra: Xác suất User thích bài hát
        return tf.nn.sigmoid(deep_out + fm_out)


# Khởi tạo Ranker với các chiều đặc trưng
# [User_ID, User_Fav, Device, Song_ID, Song_Genre, Song_Mood]
cat_dims = [NUM_USERS, NUM_GENRES, 3, NUM_SONGS, NUM_GENRES, 4]
ranker = DeepFMRankerPro(cat_dims)
ranker.compile(optimizer='adam', loss='binary_crossentropy')
