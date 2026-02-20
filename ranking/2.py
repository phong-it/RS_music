def prepare_ranking_input(user_id, candidate_indices, u_static, u_behavior, s_content):
    """
    Biến đổi ID từ Retrieval thành ma trận đầu vào cho DeepFM
    """
    ranking_data = []
    u_fav_genre = u_static[0][user_id]  # Lấy từ dữ liệu Người A
    u_dev = u_behavior[1][user_id]     # Lấy từ dữ liệu Người B

    for s_idx in candidate_indices:
        s_mood = s_content[1][s_idx]   # Lấy từ dữ liệu Người C
        s_gen = s_content[2][s_idx]    # Lấy từ dữ liệu Người C

        # Feature row: [User_ID, User_Fav, Device, Song_ID, Song_Genre, Song_Mood]
        row = [user_id, u_fav_genre, u_dev, s_idx, s_gen, s_mood]
        ranking_data.append(row)

    return np.array(ranking_data).astype('int32')


# Giả sử chúng ta lấy Top 50 từ FAISS để đưa vào Ranking
candidates_from_faiss = indices[0]  # Lấy từ bước trước của bạn
X_rank = prepare_ranking_input(user_id, candidates_from_faiss,
                               (u_fav, u_age), (u_skip, u_dev), (s_bpm, s_mood, s_genre))
