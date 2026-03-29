import pandas as pd
import numpy as np
import json
import os

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG (SYSTEM CONFIG)
# ==========================================
INPUT_FILE = 'data.csv'
OUTPUT_FILE = 'user_profiles.json'
AUDIO_FEATURES = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
ACTION_WEIGHTS = {
    'onboarding': 10,
    'follow': 7,
    'playlist': 5,
    'share': 4,
    'tag': 3,
    'like': 2,
    'dislike': -5
}


def run_pipeline():
    # --- BƯỚC 1: NẠP VÀ CHUẨN HÓA DỮ LIỆU ---
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Lỗi: Không tìm thấy {INPUT_FILE}")
        return

    df_items = pd.read_csv(INPUT_FILE)

    # Đảm bảo có cột thể loại (Genre) để xử lý "Gắn thẻ"
    if 'genre' not in df_items.columns:
        df_items['genre'] = np.random.choice(
            ['Pop', 'Rock', 'Jazz', 'Lo-fi', 'EDM'], size=len(df_items))

    # Chuẩn hóa Min-Max cho các đặc tính âm thanh (đặc biệt là Tempo)
    for feat in AUDIO_FEATURES:
        min_val = df_items[feat].min()
        max_val = df_items[feat].max()
        df_items[f'{feat}_norm'] = (
            df_items[feat] - min_val) / (max_val - min_val)

    norm_features = [f'{f}_norm' for f in AUDIO_FEATURES]

    # --- BƯỚC 2: MÔ PHỎNG DỮ LIỆU ĐẦU VÀO (INGESTION SIMULATION) ---
    # Trong thực tế, phần này sẽ thay bằng việc đọc từ Kafka hoặc Log Database
    print("⏳ Đang mô phỏng quá trình thu thập phản hồi tường minh...")
    raw_logs = []
    np.random.seed(42)
    users = [f'User_{i:02d}' for i in range(1, 11)]  # Giả lập 10 users

    for uid in users:
        for _ in range(50):  # Mỗi user 50 tương tác
            song = df_items.sample(1).iloc[0]
            raw_logs.append({
                'user_id': uid,
                'action': np.random.choice(list(ACTION_WEIGHTS.keys())),
                'id': song['id'],
                'artist_name': song['artist_name'],
                'genre': song['genre']
            })

    df_logs = pd.DataFrame(raw_logs)
    df_merged = pd.merge(df_logs, df_items[['id'] + norm_features], on='id')
    df_merged['weight'] = df_merged['action'].map(ACTION_WEIGHTS)

    # --- BƯỚC 3: XỬ LÝ TRÍCH XUẤT HỒ SƠ (ETL LOGIC) ---
    print("⚙️ Đang xử lý xây dựng Hồ sơ tĩnh (Static Profiles)...")
    final_profiles = []

    for user_id, group in df_merged.groupby('user_id'):
        # 1. Trích xuất Blacklist (Dislikes)
        blocked = list(set(group[group['action'] == 'dislike']['artist_name']))

        # 2. Lọc dữ liệu tích cực để xây dựng gu
        pos_group = group[group['weight'] > 0]
        total_w = pos_group['weight'].sum()

        profile = {
            'user_id': user_id,
            'audio_fingerprint': {},
            'metadata': {
                'top_artists': pos_group.groupby('artist_name')['weight'].sum().nlargest(3).index.tolist(),
                'top_tags': pos_group.groupby('genre')['weight'].sum().nlargest(2).index.tolist(),
                'blocked_artists': blocked
            }
        }

        # 3. Tính toán Weighted Average cho Audio Features
        if total_w > 0:
            for orig_f, norm_f in zip(AUDIO_FEATURES, norm_features):
                weighted_avg = (pos_group[norm_f] *
                                pos_group['weight']).sum() / total_w
                profile['audio_fingerprint'][orig_f] = round(
                    float(weighted_avg), 4)

        final_profiles.append(profile)

    # --- BƯỚC 4: XUẤT KẾT QUẢ (DATA OUTPUT) ---
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_profiles, f, indent=4, ensure_ascii=False)

    print(f"✅ Thành công! Đã tạo hồ sơ cho {len(final_profiles)} người dùng.")
    print(f"📁 Kết quả lưu tại: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    run_pipeline()
