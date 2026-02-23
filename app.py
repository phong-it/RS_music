import streamlit as st
import pandas as pd
import numpy as np
import ast

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.set_page_config(page_title="User Profile System", layout="wide")
st.title("🎧 Hệ thống Trích xuất Hồ sơ Người dùng (Audio Profile)")
st.markdown("""
Hệ thống sử dụng kiến trúc **Cascade + Mixed** để lọc (Blacklist) và tổng hợp (Weighted Average) 
các đặc trưng âm thanh từ lịch sử tương tác của người dùng.
""")

# Cột cấu hình
st.sidebar.header("⚙️ Cấu hình hệ thống")
num_users = st.sidebar.slider("Số lượng User mô phỏng:", 1, 10, 3)
num_interactions = st.sidebar.slider("Số tương tác mỗi User:", 10, 100, 30)

# ==========================================
# UPLOAD VÀ XỬ LÝ DỮ LIỆU
# ==========================================
uploaded_file = st.file_uploader("📂 Tải lên dataset của bạn (data.csv)", type=["csv"])

if uploaded_file is not None:
    # 1. Đọc dữ liệu
    df_items = pd.read_csv(uploaded_file)
    st.success(f"Đã tải thành công file với {len(df_items)} bài hát!")
    
    with st.expander("👀 Xem trước dữ liệu bài hát (Item Metadata)"):
        st.dataframe(df_items.head(5))

    # 2. Xử lý khi bấm nút
    if st.button("🚀 Chạy Mô Phỏng Tương Tác & Sinh Hồ Sơ"):
        with st.spinner("Đang chạy thuật toán Cascade + Mixed..."):
            
            # --- TẠO MOCK DATA TƯƠNG TÁC ---
            np.random.seed(42)
            users = [f'user_{i:02d}' for i in range(1, num_users + 1)]
            actions = ['onboarding', 'follow', 'playlist', 'share', 'like', 'dislike']
            action_probs = [0.1, 0.15, 0.2, 0.15, 0.3, 0.1]

            interactions = []
            for user in users:
                for _ in range(num_interactions):
                    idx = np.random.randint(0, len(df_items))
                    interactions.append({
                        'user_id': user,
                        'action': np.random.choice(actions, p=action_probs),
                        'id': df_items.loc[idx, 'id'] # Giả sử file có cột 'id'
                    })
            df_logs = pd.DataFrame(interactions)
            
            # --- JOIN LOGS VÀ ITEMS ---
            df_merged = pd.merge(df_logs, df_items, on='id')
            
            # --- THUẬT TOÁN CASCADE + MIXED ---
            weights = {'onboarding': 10, 'follow': 7, 'playlist': 5, 'share': 4, 'like': 2, 'dislike': -1}
            audio_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
            
            df_merged['score'] = df_merged['action'].map(weights).fillna(0)
            
            # Cascade
            dislikes = df_merged[df_merged['action'] == 'dislike']
            blocked_artists = dislikes.groupby('user_id')['artist_name'].apply(set).to_dict()
            valid_df = df_merged[df_merged['action'] != 'dislike'].copy()

            # Mixed
            profiles = []
            for user_id, group in valid_df.groupby('user_id'):
                blocked = blocked_artists.get(user_id, set())
                group = group[~group['artist_name'].isin(blocked)]
                
                total_score = group['score'].sum()
                user_audio_profile = {}
                
                if total_score > 0:
                    for feature in audio_features:
                        if feature in group.columns: # Kiểm tra xem feature có trong file csv không
                            weighted_sum = (group[feature] * group['score']).sum()
                            user_audio_profile[feature] = round(weighted_sum / total_score, 3)
                
                top_artists = group.groupby('artist_name')['score'].sum().sort_values(ascending=False).head(5).index.tolist()
                
                profiles.append({
                    'user_id': user_id,
                    'favorite_artists': top_artists,
                    'audio_fingerprint': user_audio_profile,
                    'blocked_artists': list(blocked)
                })

            # --- HIỂN THỊ KẾT QUẢ RA GIAO DIỆN ---
            st.markdown("### 🎯 Kết quả: Hồ sơ tĩnh (Static User Profiles)")
            
            # Dùng Tabs để hiển thị từng user cho gọn gàng
            tabs = st.tabs([p['user_id'] for p in profiles])
            
            for idx, tab in enumerate(tabs):
                with tab:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("⭐ **Top Nghệ sĩ yêu thích:**")
                        for artist in profiles[idx]['favorite_artists']:
                            st.write(f"- {artist}")
                            
                        st.write("⛔ **Danh sách chặn (Blacklist):**")
                        if profiles[idx]['blocked_artists']:
                            for artist in profiles[idx]['blocked_artists']:
                                st.error(f"- {artist}")
                        else:
                            st.write("- Trống")
                            
                    with col2:
                        st.write("🎶 **Đặc trưng Âm thanh (Audio Fingerprint):**")
                        st.json(profiles[idx]['audio_fingerprint'])

            # Nút tải xuống file JSON
            import json
            json_string = json.dumps(profiles, indent=4, ensure_ascii=False)
            st.download_button(
                label="📥 Tải xuống dữ liệu Profiles (JSON)",
                file_name="user_audio_profiles.json",
                mime="application/json",
                data=json_string,
            )
else:
    st.info("Vui lòng tải lên file data.csv của bạn để bắt đầu!")