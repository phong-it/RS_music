# KHẮC PHỤC LỖI TRONG ẢNH: Đảm bảo đã cài đặt faiss
# !pip install faiss-cpu

import faiss

# 1. Dự đoán điểm số bằng mô hình Pro (nhận 2 loại input)
scores = ranker.predict([X_rank_cat, X_rank_num])

# 2. Sắp xếp kết quả tinh hoa
final_ranked = sorted(zip(candidates_faiss, scores.flatten()),
                      key=lambda x: x[1], reverse=True)

print(f"\n--- DANH SÁCH 'CHÂN ÁI' CHO USER {user_id} (DEEPFM PRO) ---")
for i, (s_idx, score) in enumerate(final_ranked[:5]):
    # Lấy thông tin thực tế để kiểm chứng
    actual_genre = genres[int(s_genre[s_idx])]
    print(
        f"Hạng {i+1}: Song ID {s_idx:4} | Xác suất thích: {score:.4f} | Thể loại: {actual_genre}")
