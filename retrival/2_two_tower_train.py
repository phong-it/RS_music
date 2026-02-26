import tensorflow as tf
from tensorflow.keras import layers, Model

EMBEDDING_DIM = 32


class MultiInputTower(layers.Layer):
    """Xử lý song song dữ liệu ID (Embedding) và dữ liệu số"""

    def __init__(self, vocab_sizes, embed_size):
        super().__init__()
        self.embeddings = [layers.Embedding(
            v, embed_size) for v in vocab_sizes]
        self.dense_net = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(EMBEDDING_DIM)
        ])

    def call(self, cat_inputs, num_inputs):
        # Biến ID thành Vector có ý nghĩa
        embed_outs = [self.embeddings[i](cat_inputs[:, i])
                      for i in range(len(self.embeddings))]
        # Nối tất cả lại trước khi đưa vào mạng Deep
        combined = layers.Concatenate()([*embed_outs, num_inputs])
        return self.dense_net(combined)


class ProTwoTower(Model):
    def __init__(self):
        super().__init__()
        # User: Genre(6), Device(3) | Song: Genre(6), Mood(4)
        self.user_tower = MultiInputTower([6, 3], 8)
        self.song_tower = MultiInputTower([6, 4], 8)

    def call(self, inputs):
        (u_cat, u_num), (s_cat, s_num) = inputs
        u_emb = tf.nn.l2_normalize(self.user_tower(u_cat, u_num), axis=1)
        s_emb = tf.nn.l2_normalize(self.song_tower(s_cat, s_num), axis=1)
        return u_emb, s_emb


model = ProTwoTower()
optimizer = tf.keras.optimizers.Adam(0.001)

# Huấn luyện thử nghiệm (Dùng 100 mẫu đầu tiên)
for epoch in range(11):
    with tf.GradientTape() as tape:
        u_emb, s_emb = model([(u_cat_features[:100], u_num_features[:100]),
                              (s_cat_features[:100], s_num_features[:100])])
        logits = tf.matmul(u_emb, s_emb, transpose_b=True) / 0.1
        labels = tf.range(100)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
