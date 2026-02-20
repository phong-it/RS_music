# --- KIẾN TRÚC MÔ HÌNH ---
class TwoTowerModel(Model):
    def __init__(self, user_dim, song_dim):
        super().__init__()
        # Tháp User
        self.user_tower = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(EMBEDDING_DIM)
        ])
        # Tháp Song
        self.song_tower = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(EMBEDDING_DIM)
        ])

    def call(self, inputs):
        user_feat, song_feat = inputs
        u_emb = tf.nn.l2_normalize(self.user_tower(user_feat), axis=1)
        s_emb = tf.nn.l2_normalize(self.song_tower(song_feat), axis=1)
        return u_emb, s_emb


model = TwoTowerModel(user_features.shape[1], song_features.shape[1])
optimizer = tf.keras.optimizers.Adam(0.01)

# Huấn luyện nhanh
for epoch in range(50):
    with tf.GradientTape() as tape:
        # Giả sử trong batch này, User i nghe Song i (dữ liệu tương tác thật)
        u_emb, s_emb = model([user_features[:100], song_features[:100]])
        # Tính Dot Product để tìm sự tương đồng
        # 0.1 là temperature để làm sắc nét phân phối
        logits = tf.matmul(u_emb, s_emb, transpose_b=True) / 0.1
        labels = tf.range(100)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
