# 1. Đọc bảng tương tác thật
df_inter = pd.read_csv('user_interactions.csv')  # Đây là lúc nó xuất hiện!

# 2. Lấy danh sách ID đã tương tác (ví dụ 100 tương tác đầu tiên)
inter_batch = df_inter.head(100)
u_ids = inter_batch['user_id'].values
s_ids = inter_batch['song_id'].values

# 3. Lấy đặc trưng tương ứng từ các mảng bạn đã chuẩn bị ở File 1
u_cat_batch = u_cat_features[u_ids]
u_num_batch = u_num_features[u_ids]

s_cat_batch = s_cat_features[s_ids]
s_num_batch = s_num_features[s_ids]

# 4. Đưa vào huấn luyện (Lúc này labels = tf.range(100) là hoàn toàn chính xác)
with tf.GradientTape() as tape:
    u_emb, s_emb = model(
        [(u_cat_batch, u_num_batch), (s_cat_batch, s_num_batch)])
    logits = tf.matmul(u_emb, s_emb, transpose_b=True) / 0.1
    labels = tf.range(100)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
