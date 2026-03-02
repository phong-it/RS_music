import faiss
import numpy as np

# 1. Tạo Index cho kho nhạc (Thực hiện một lần hoặc khi kho nhạc có bài mới)
# model.song_tower nhận (cat, num) từ File 2
all_song_embeddings = model.song_tower(s_cat_features, s_num_features).numpy()

# Sử dụng IndexFlatIP cho tích vô hướng (Inner Product) - phù hợp với l2_normalize
faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
faiss_index.add(all_song_embeddings)

# 2. Truy vấn thực tế cho User X
# Giả sử user_id được chọn từ trước (ví dụ: user_id = 42)
u_cat_query = u_cat_features[user_id:user_id+1]
u_num_query = u_num_features[user_id:user_id+1]

# Lấy vector đại diện của User từ tháp (User Tower)
u_vector = model.user_tower(u_cat_query, u_num_query).numpy()

# 3. FAISS quét toàn bộ kho bài hát
# Lấy TOP_K đủ lớn (ví dụ 50) để làm "đầu vào thô" cho tầng Ranking phía sau
TOP_K_RETRIEVAL = 50
distances, indices = faiss_index.search(u_vector, TOP_K_RETRIEVAL)

# 4. Lưu lại danh sách ứng viên để File 5 và 6 sử dụng
candidate_list = indices[0]

print(
    f"File 3: Đã lọc thô {TOP_K_RETRIEVAL} bài hát tiềm năng cho User {user_id}")
print(f"Top 3 ID ứng viên đầu tiên: {candidate_list[:3]}")

# --- KIỂM CHỨNG NHANH (Optional) ---
for i in range(3):
    s_idx = candidate_list[i]
    # Lấy tên thể loại thực tế từ encoder ở File 1
    genre_name = genre_encoder.inverse_transform(
        [int(s_cat_features[s_idx, 0])])[0]
    print(
        f"  - Ứng viên {i+1}: ID {s_idx} | Genre: {genre_name} | Score: {distances[0][i]:.4f}")
