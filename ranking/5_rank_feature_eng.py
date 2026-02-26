def prepare_pro_ranking_input(user_id, candidate_indices, u_static, u_behavior, s_content):
    """
    Chuẩn bị dữ liệu cho Ranking bao gồm cả ID và đặc trưng số
    """
    cat_data = []
    num_data = []

    # Lấy dữ liệu User (A + B)
    u_fav_genre = u_static[0][user_id]
    u_dev = u_behavior[1][user_id]
    u_num = u_num_features[user_id]  # Dữ liệu số đã scale (Age, Skip Rate)

    for s_idx in candidate_indices:
        # Lấy dữ liệu bài hát (C)
        s_mood = s_content[1][s_idx]
        s_gen = s_content[2][s_idx]
        s_num = s_num_features[s_idx]  # Dữ liệu số đã scale (BPM)

        # 1. Nhóm Categorical: [User_ID, Fav_Gen, Dev, Song_ID, Song_Gen, Mood]
        cat_row = [user_id, u_fav_genre, u_dev, s_idx, s_gen, s_mood]
        # 2. Nhóm Numerical: [User_Age_Scaled, User_Skip_Scaled, Song_BPM_Scaled]
        num_row = np.concatenate([u_num, s_num])

        cat_data.append(cat_row)
        num_data.append(num_row)

    return np.array(cat_data).astype('int32'), np.array(num_data).astype('float32')


# Tạo input từ Top 50 của FAISS
candidates_faiss = indices[0]
X_rank_cat, X_rank_num = prepare_pro_ranking_input(
    user_id, candidates_faiss, (u_fav, u_age), (u_skip,
                                                u_dev), (s_bpm, s_mood, s_genre)
)
