import numpy as np


def prepare_ranking_input_pro(user_id, candidate_indices):
    """
    Biến đổi các ID ứng viên từ FAISS thành dữ liệu đầu vào chuẩn cho DeepFM.
    Sử dụng trực tiếp các mảng tính năng đã được chuẩn hóa từ File 1.
    """
    cat_data = []
    num_data = []

    # 1. Lấy dữ liệu của User (Cố định cho toàn bộ danh sách ứng viên)
    # u_cat_features: [Genre_ID, Device_ID]
    # u_num_features: [Age_Scaled, Skip_Rate_Scaled]
    u_cat = u_cat_features[user_id]
    u_num = u_num_features[user_id]

    # 2. Duyệt qua từng ứng viên bài hát được FAISS gợi ý
    for s_idx in candidate_indices:
        # Lấy dữ liệu của bài hát s_idx
        # s_cat_features: [Genre_ID, Mood_ID]
        # s_num_features: [BPM_Scaled]
        s_cat = s_cat_features[s_idx]
        s_num = s_num_features[s_idx]

        # --- TẠO VECTOR PHÂN LOẠI (CATEGORICAL) ---
        # Cấu trúc: [User_ID, User_Genre, User_Device, Song_ID, Song_Genre, Song_Mood]
        # Lưu ý: Bao gồm cả ID để mô hình học đặc tính riêng biệt của từng User/Song
        cat_row = [user_id, u_cat[0], u_cat[1], s_idx, s_cat[0], s_cat[1]]

        # --- TẠO VECTOR SỐ (NUMERICAL) ---
        # Cấu trúc: [Age_Scaled, Skip_Rate_Scaled, BPM_Scaled]
        num_row = np.concatenate([u_num, s_num])

        cat_data.append(cat_row)
        num_data.append(num_row)

    # Chuyển thành mảng Numpy để đưa vào mô hình AI
    X_rank_cat = np.array(cat_data).astype('int32')
    X_rank_num = np.array(num_data).astype('float32')

    return X_rank_cat, X_rank_num


# --- THỰC THI THỬ NGHIỆM ---
# Giả sử 'indices' là kết quả trả về từ File 3 (FAISS)
if 'indices' in locals():
    candidate_list = indices[0]  # Lấy danh sách ID bài hát ứng viên
    X_rank_cat, X_rank_num = prepare_ranking_input_pro(user_id, candidate_list)

    print(f"File 5: Đã chuẩn bị dữ liệu Ranking cho User {user_id}")
    print(f" - Hình dạng đầu vào ID (Categorical): {X_rank_cat.shape}")
    print(f" - Hình dạng đầu vào Số (Numerical): {X_rank_num.shape}")
else:
    print("Cảnh báo: Không tìm thấy biến 'indices'. Hãy đảm bảo File 3 đã chạy thành công.")
