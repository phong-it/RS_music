# 1. Dự đoán điểm số cho các ứng viên
scores = ranker.predict(X_rank)

# 2. Sắp xếp lại dựa trên điểm số của DeepFM
ranked_candidates = sorted(zip(candidates_from_faiss, scores.flatten()),
                           key=lambda x: x[1], reverse=True)

print(f"\n--- KẾT QUẢ XẾP HẠNG CUỐI CÙNG (DEEPFM) CHO USER {user_id} ---")
for i, (s_idx, score) in enumerate(ranked_candidates[:5]):
    print(f"Hạng {i+1}: Song ID {s_idx} | Ranking Score: {score:.4f}")
    # Kiểm tra lại xem có khớp gu Jazz của User không
    print(f"   -> Thể loại thực tế: {genres[int(s_genre[s_idx])]}")
