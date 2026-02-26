import faiss

# 1. Tạo Index cho kho nhạc (Offline)
# Truyền đồng thời dữ liệu phân loại và dữ liệu số của bài hát
all_song_embeddings = model.song_tower(s_cat_features, s_num_features).numpy()
faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
faiss_index.add(all_song_embeddings)

# 2. Truy vấn thực tế cho User 42
user_id = 42
u_cat_query = u_cat_features[user_id:user_id+1]
u_num_query = u_num_features[user_id:user_id+1]

# Lấy vector đại diện của User từ tháp đa đầu vào
u_vector = model.user_tower(u_cat_query, u_num_query).numpy()

# FAISS quét toàn bộ kho nhạc lấy Top 5
distances, indices = faiss_index.search(u_vector, 5)

print(f"\n--- KẾT QUẢ ĐỀ XUẤT HOÀN HẢO CHO USER {user_id} ---")
for i in range(5):
    s_idx = indices[0][i]
    print(f"Top {i+1}: Song ID {s_idx} | Match Score: {distances[0][i]:.4f}")
    # Hiển thị đặc trưng thực tế để kiểm chứng độ khớp
    print(
        f"   -> Genre ID: {s_cat_features[s_idx, 0]} | BPM: {s_bpm[s_idx][0]}")
