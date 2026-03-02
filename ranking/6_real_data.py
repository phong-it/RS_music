
# !pip install faiss-cpu

import faiss

# 1. Dự đoán điểm số cho các ứng viên (Đã chuẩn bị từ File 5)
# X_rank_cat, X_rank_num là dữ liệu của Top 50 bài hát ứng viên
scores = ranker.predict([X_rank_cat, X_rank_num], verbose=0)

# 2. Hợp nhất ID bài hát và điểm số để sắp xếp
# candidates_faiss lấy từ File 3 (indices[0])
final_results = []
for i, s_idx in enumerate(candidate_list):
    final_results.append({
        'song_id': s_idx,
        'score': float(scores[i]),
        'genre_id': int(s_cat_features[s_idx, 0])
    })

# Sắp xếp từ cao xuống thấp theo Score
final_ranked = sorted(final_results, key=lambda x: x['score'], reverse=True)

# 3. Hiển thị kết quả chuyên nghiệp
print(f"\n" + "="*50)
print(f"DANH SÁCH GỢI Ý TỐI ƯU CHO USER: {user_id}")
print("="*50)

for i, item in enumerate(final_ranked[:5]):
    # Dùng encoder từ File 1 để dịch ngược ID sang tên thể loại
    genre_name = genre_encoder.inverse_transform([item['genre_id']])[0]

    print(f"Hạng {i+1}:")
    print(f"   - Mã bài hát: {item['song_id']}")
    print(f"   - Độ phù hợp: {item['score']*100:.2f}%")
    print(f"   - Thể loại:   {genre_name}")
    print("-" * 20)
