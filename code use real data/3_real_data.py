# 0. Cài đặt thư viện (Chỉ chạy một lần)
# !pip install faiss-cpu

import faiss
import numpy as np
import tensorflow as tf

# 1. Tạo Index cho kho nhạc (Ép kiểu float32 là bắt buộc cho FAISS)
# model.song_tower sẽ trả về vector đặc trưng của toàn bộ bài hát
all_song_embeddings = model.song_tower(
    s_cat_features, s_num_features).numpy().astype('float32')

# Khởi tạo Index với chiều tương ứng (EMBEDDING_DIM = 32)
faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
faiss_index.add(all_song_embeddings)

# 2. Truy vấn thực tế cho User X
user_id = 0  # Đã sửa lỗi NameError bằng cách định nghĩa giá trị ở đây

# Lấy đặc trưng của người dùng dựa trên vị trí index
u_cat_query = u_cat_features[user_id: user_id + 1]
u_num_query = u_num_features[user_id: user_id + 1]

# Chuyển đổi thông tin người dùng thành Vector (User Embedding)
u_vector = model.user_tower(u_cat_query, u_num_query).numpy().astype('float32')

# 3. FAISS tìm kiếm TOP_K bài hát có độ tương đồng cao nhất
TOP_K_RETRIEVAL = 3  # Kho nhạc có 5 bài, nên lấy Top 3 là hợp lý
distances, indices = faiss_index.search(u_vector, TOP_K_RETRIEVAL)

# 4. Hiển thị kết quả gợi ý
candidate_list = indices[0]
print(f"--- Đã lọc được {len(candidate_list)} bài hát cho User {user_id} ---")

for i in range(len(candidate_list)):
    s_idx = candidate_list[i]
    # Lấy tên thể loại từ encoder đã chuẩn bị ở File 1
    genre_idx = int(s_cat_features[s_idx, 0])
    genre_name = genre_encoder.inverse_transform([genre_idx])[0]

    print(
        f"Top {i+1}: Bài hát Index {s_idx} | Thể loại: {genre_name} | Điểm tương đồng: {distances[0][i]:.4f}")
