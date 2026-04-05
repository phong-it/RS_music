import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

EMBEDDING_DIM = 32


class MultiInputTower(layers.Layer):
    """Tháp xử lý độc lập cho User hoặc Song"""

    def __init__(self, vocab_sizes, embed_size=8):
        super().__init__()
        # Kích thước từ vựng (+1 để dự phòng các giá trị unseen nếu có)
        self.embeddings = [layers.Embedding(input_dim=v + 1, output_dim=embed_size)
                           for v in vocab_sizes]

        self.dense_net = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(EMBEDDING_DIM)
        ])

    def call(self, cat_inputs, num_inputs):
        # Duyệt qua từng cột Categorical để nhúng (Embed)
        embed_outs = [self.embeddings[i](cat_inputs[:, i])
                      for i in range(len(self.embeddings))]

        # Gộp tất cả Categorical lại
        combined_cat = layers.Concatenate()(embed_outs) if embed_outs else None

        # Gộp Categorical và Numerical
        if combined_cat is not None:
            combined = layers.Concatenate()(
                [combined_cat, tf.cast(num_inputs, tf.float32)])
        else:
            combined = tf.cast(num_inputs, tf.float32)

        return self.dense_net(combined)


class ProTwoTower(Model):
    """Mô hình Two-Tower dự đoán Engagement Score bằng Dot Product"""

    def __init__(self, user_vocab_sizes, song_vocab_sizes):
        super().__init__()
        # Khởi tạo kích thước động thay vì hardcode
        self.user_tower = MultiInputTower(user_vocab_sizes)
        self.song_tower = MultiInputTower(song_vocab_sizes)

    def call(self, inputs):
        # Giải nén 4 cục data
        u_cat, u_num, s_cat, s_num = inputs

        # Lấy Embeddings và chuẩn hóa L2
        u_emb = tf.nn.l2_normalize(self.user_tower(u_cat, u_num), axis=1)
        s_emb = tf.nn.l2_normalize(self.song_tower(s_cat, s_num), axis=1)

        # Pointwise Learning: Tính Tích vô hướng (Dot Product)
        # Vì đã L2 Normalize, kết quả sẽ nằm trong khoảng [-1, 1] (Cosine Similarity)
        dot_product = tf.reduce_sum(tf.multiply(
            u_emb, s_emb), axis=1, keepdims=True)
        return dot_product


class TwoTowerTrainer:
    """Class quản lý toàn bộ vòng đời huấn luyện"""

    def __init__(self, user_vocab_sizes, song_vocab_sizes, learning_rate=0.001):
        self.model = ProTwoTower(user_vocab_sizes, song_vocab_sizes)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Sử dụng MSE (Mean Squared Error) vì y là engagement_score (số thực)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def create_dataset(self, u_cat, u_num, s_cat, s_num, y, batch_size=64):
        """Chuyển đổi Numpy arrays thành tf.data.Dataset để chống lệch dòng"""
        dataset = tf.data.Dataset.from_tensor_slices((
            (u_cat, u_num, s_cat, s_num),
            tf.cast(y, tf.float32)
        ))
        # Xáo trộn và chia lô tự động tối ưu RAM
        dataset = dataset.shuffle(buffer_size=1000).batch(
            batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    @tf.function  # Decorator này giúp biên dịch thành Graph C++, tăng tốc độ train 2-3 lần
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        return loss

    def fit(self, dataset, epochs=10):
        print("Bắt đầu quá trình huấn luyện mạng Two-Tower...")
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for inputs, labels in dataset:
                loss = self.train_step(inputs, labels)
                total_loss += loss
                num_batches += 1

            print(
                f"   > Epoch {epoch + 1:02d}/{epochs} - Loss (MSE): {total_loss/num_batches:.4f}")
        print("Huấn luyện hoàn tất!")


# HƯỚNG DẪN KẾT NỐI THỰC TẾ VỚI FILE 1 (X_cat, X_num, y)
if __name__ == "__main__":

    # tạo data giả lập để test code không bị lỗi
    num_samples = 150
    u_cat_mock = np.random.randint(0, 5, size=(num_samples, 2))  # 2 features
    s_cat_mock = np.random.randint(0, 5, size=(num_samples, 2))  # 2 features
    u_num_mock = np.random.rand(num_samples, 4)  # 4 features
    s_num_mock = np.random.rand(num_samples, 2)  # 2 features
    y_mock = np.random.rand(num_samples, 1)  # Engagement score

    # 2. Truyền động kích thước từ vựng (Dynamic Vocab)
    # Tìm giá trị lớn nhất trong mỗi cột categorical + 1
    user_vocabs = [np.max(u_cat_mock[:, i]) +
                   1 for i in range(u_cat_mock.shape[1])]
    song_vocabs = [np.max(s_cat_mock[:, i]) +
                   1 for i in range(s_cat_mock.shape[1])]

    # 3. Khởi tạo và chạy
    trainer = TwoTowerTrainer(
        user_vocab_sizes=user_vocabs, song_vocab_sizes=song_vocabs)

    train_dataset = trainer.create_dataset(
        u_cat_mock, u_num_mock,
        s_cat_mock, s_num_mock,
        y_mock,
        batch_size=32
    )

    trainer.fit(train_dataset, epochs=5)
