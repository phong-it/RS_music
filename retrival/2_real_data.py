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


# 1. Đọc bảng tương tác thật
df_inter = pd.read_csv('user_interactions.csv')  # Đây là lúc nó xuất hiện!

# 2. Lấy danh sách ID đã tương tác (ví dụ 100 tương tác đầu tiên)
inter_batch = df_inter.head(100)
# 1. Tạo "cuốn từ điển" để máy biết ID 101 tương ứng với hàng 0
song_id_map = {id: idx for idx, id in enumerate(df_songs_c['song_id'].values)}
user_id_map = {id: idx for idx, id in enumerate(df_users['user_id'].values)}

# 2. Lấy danh sách ID từ bảng tương tác
u_ids_raw = inter_batch['user_id'].values
s_ids_raw = inter_batch['song_id'].values

# 3. Chuyển đổi ID thực (101, 102...) sang vị trí Index (0, 1...)
# .get(id) giúp tránh lỗi nếu bạn lỡ nhập một ID không tồn tại trong kho
u_indices = [user_id_map.get(uid) for uid in u_ids_raw if uid in user_id_map]
s_indices = [song_id_map.get(sid) for sid in s_ids_raw if sid in song_id_map]

# 4. Truy xuất đặc trưng bằng Index đã được ánh xạ
u_cat_batch = u_cat_features[u_indices]
u_num_batch = u_num_features[u_indices]

s_cat_batch = s_cat_features[s_indices]
s_num_batch = s_num_features[s_indices]

# --- PHẦN SỬA LỖI VALUEERROR ---

with tf.GradientTape() as tape:
    # 1. Dự đoán Vector đặc trưng (Embeddings)
    u_emb, s_emb = model(
        [(u_cat_batch, u_num_batch), (s_cat_batch, s_num_batch)])

    # 2. Tính toán ma trận tương quan (Logits)
    logits = tf.matmul(u_emb, s_emb, transpose_b=True) / 0.1

    # 3. SỬA TẠI ĐÂY: Lấy kích thước thực tế của lô dữ liệu hiện tại
    current_batch_size = tf.shape(u_emb)[0]
    labels = tf.range(current_batch_size)  # Thay vì tf.range(100)

    # 4. Tính toán Loss
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

    # Cập nhật trọng số mô hình
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 5. SỬA TẠI ĐÂY: Đảm bảo biến epoch đã được định nghĩa (ví dụ trong vòng lặp)
    # Nếu bạn chưa có vòng lặp, hãy xóa điều kiện if epoch % 5 == 0:
    print(f"Huấn luyện thành công! Loss hiện tại: {loss.numpy():.4f}")
