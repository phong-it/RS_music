class DeepFMRanker(Model):
    def __init__(self, feature_dims, embedding_dim=8):
        super(DeepFMRanker, self).__init__()
        # feature_dims: Danh sách số lượng nhãn cho mỗi feature (ví dụ: [1000, 6, 4, 3])

        # 1. Embedding layer cho từng đặc trưng (Dùng chung cho cả FM và Deep)
        self.embeddings = [layers.Embedding(input_dim=dim, output_dim=embedding_dim)
                           for dim in feature_dims]

        # 2. Tuyến Deep (DNN)
        self.dnn = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Đầu ra tuyến Deep
        ])

        # 3. Tuyến FM (Học tương tác bậc 2)
        # Đơn giản hóa: Cộng các tương tác Linear
        self.linear = layers.Dense(1)

    def call(self, inputs):
        # inputs: [Batch, Num_Features]
        # Tách các feature và đưa qua embedding
        embedded = [self.embeddings[i](inputs[:, i])
                    for i in range(len(self.embeddings))]

        # Tính toán phần Deep
        deep_out = self.dnn(tf.stack(embedded, axis=1))

        # Tính toán phần FM (Linear part)
        fm_out = self.linear(tf.reduce_sum(tf.stack(embedded, axis=1), axis=1))

        # Kết quả cuối cùng là xác suất User thích bài hát
        return tf.nn.sigmoid(deep_out + fm_out)


# --- KHỞI TẠO RANKER ---
# Đặc trưng chọn lọc: [User_ID, User_Fav_Genre, Device, Song_ID, Song_Genre, Song_Mood]
feature_dims = [NUM_USERS, len(genres), 3, NUM_SONGS, len(genres), 4]
ranker = DeepFMRanker(feature_dims)
ranker.compile(optimizer='adam', loss='binary_crossentropy')
