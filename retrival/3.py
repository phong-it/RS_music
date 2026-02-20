# 1. Tạo Index cho kho nhạc (Offline)
all_song_embeddings = model.song_tower(song_features).numpy()
faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
faiss_index.add(all_song_embeddings)

# 2. Truy vấn thực tế cho User X
user_id = 42
# Giả sử lúc này Người C báo thêm: User đang ở trạng thái 'Chill'
# Chúng ta có thể cập nhật nhẹ profile user trước khi đưa vào tháp
current_user_vector = user_features[user_id:user_id+1]

# Lấy vector đại diện của User từ tháp
u_vector = model.user_tower(current_user_vector).numpy()

# FAISS quét toàn bộ kho nhạc
top_k = 5
distances, indices = faiss_index.search(u_vector, top_k)

print(f"\n--- KẾT QUẢ ĐỀ XUẤT CHO USER {user_id} ---")
for i in range(top_k):
    s_idx = indices[0][i]
    print(f"Top {i+1}: Song ID {s_idx} | Match Score: {distances[0][i]:.4f}")
    print(f"   -> Đặc trưng bài hát (BPM/Mood/Genre): {song_features[s_idx]}")
