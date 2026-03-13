import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================
st.set_page_config(page_title="Music User Profile & RecSys", layout="wide")
st.title("🎧 Hệ thống Hồ sơ Người dùng & Đề xuất Âm nhạc")
st.markdown("""
Hệ thống thực hiện quy trình: **Thu thập phản hồi tường minh** ➔ **Xây dựng Hồ sơ tĩnh (Static Profile)** ➔ **Đề xuất dựa trên nội dung (Content-based Filtering)**.
""")

# ==========================================
# 2. ĐỌC DỮ LIỆU GỐC
# ==========================================
try:
    df_items = pd.read_csv('data.csv')
    audio_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
except:
    st.error("Lỗi: Không tìm thấy file 'data.csv'. Vui lòng đảm bảo file csv nằm cùng thư mục với code.")
    st.stop()

# ==========================================
# 3. THUẬT TOÁN ĐỀ XUẤT (RECOMMENDATION LOGIC)
# ==========================================
def get_recommendations(user_profile, df_items, audio_features, top_n=5):
    # Trích xuất vector của User (Dấu vân tay âm thanh)
    user_vector = np.array([user_profile[f] for f in audio_features]).reshape(1, -1)
    
    # Trích xuất vector của toàn bộ kho nhạc
    item_vectors = df_items[audio_features].values
    
    # Tính độ tương đồng Cosine (Cosine Similarity)
    scores = cosine_similarity(user_vector, item_vectors).flatten()
    
    df_res = df_items.copy()
    df_res['match_score'] = np.round(scores * 100, 2) 
    
    # Lọc Blacklist (Cascade Filtering)
    blocked = user_profile.get('blocked_artists', [])
    df_res = df_res[~df_res['artist_name'].isin(blocked)]
    
    return df_res.sort_values(by='match_score', ascending=False).head(top_n)

# ==========================================
# 4. SIDEBAR - CÀI ĐẶT GIẢ LẬP
# ==========================================
st.sidebar.header("🛠 Cấu hình Giả lập")
num_users = st.sidebar.slider("Số lượng User:", 1, 15, 5)
interactions_per_user = st.sidebar.slider("Số hành động/User:", 10, 100, 30)

if st.sidebar.button("🚀 Chạy Hệ Thống"):
    # --- BƯỚC 1: GIẢ LẬP HÀNH VI NGƯỜI DÙNG (SIMULATION) ---
    actions = ['onboarding', 'follow', 'playlist', 'share', 'like', 'dislike']
    action_probs = [0.1, 0.1, 0.15, 0.15, 0.4, 0.1]
    weights = {'onboarding': 10, 'follow': 7, 'playlist': 5, 'share': 4, 'like': 2, 'dislike': -1}

    raw_logs = []
    np.random.seed(42)
    for i in range(1, num_users + 1):
        uid = f'User_{i:02d}'
        for _ in range(interactions_per_user):
            song = df_items.sample(1).iloc[0]
            raw_logs.append({
                'user_id': uid, 
                'action': np.random.choice(actions, p=action_probs), 
                'id': song['id']
            })
    
    df_interactions = pd.DataFrame(raw_logs)
    df_merged = pd.merge(df_interactions, df_items, on='id')
    df_merged['score'] = df_merged['action'].map(weights)

    # --- BƯỚC 2: XỬ LÝ HỒ SƠ NGƯỜI DÙNG (DATA ENGINEERING) ---
    profiles_data = []
    for user_id, group in df_merged.groupby('user_id'):
        # Cascade: Lọc danh sách chặn
        disliked_artists = set(group[group['action'] == 'dislike']['artist_name'])
        valid_int = group[~group['artist_name'].isin(disliked_artists) & (group['action'] != 'dislike')]
        
        total_score = valid_int['score'].sum()
        profile_stats = {'user_id': user_id, 'blocked_artists': list(disliked_artists)}
        
        if total_score > 0:
            for feature in audio_features:
                # Mixed: Tính trung bình trọng số (Weighted Average)
                weighted_sum = (valid_int[feature] * valid_int['score']).sum()
                profile_stats[feature] = round(weighted_sum / total_score, 3)
            
            profiles_data.append(profile_stats)

    df_final = pd.DataFrame(profiles_data)

    # --- BƯỚC 3: HIỂN THỊ KẾT QUẢ (VISUALIZATION) ---
    
    # Tab 1: Tổng quan Profile
    st.subheader("🎯 Bảng tổng hợp Hồ sơ tĩnh (Static Profiles)")
    st.dataframe(df_final.drop(columns=['blocked_artists']).style.highlight_max(axis=0), use_container_width=True)

    st.divider()
    
    # Biểu đồ Radar/Bar so sánh
    st.subheader("📈 Phân tích dấu vân tay âm thanh (Audio Fingerprint)")
    fig = px.bar(df_final, x='user_id', y=audio_features, barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- BƯỚC 4: HỆ THỐNG ĐỀ XUẤT (RECOMMENDATIONS) ---
    st.divider()
    st.subheader("🎵 Đề xuất bài hát cá nhân hóa")
    
    user_tabs = st.tabs([p['user_id'] for p in profiles_data])
    for idx, tab in enumerate(user_tabs):
        with tab:
            u_prof = profiles_data[idx]
            rec_results = get_recommendations(u_prof, df_items, audio_features)
            
            c1, c2 = st.columns([3, 2])
            with c1:
                st.write(f"Top 5 gợi ý cho **{u_prof['user_id']}**:")
                st.table(rec_results[['track_name', 'artist_name', 'match_score']])
            with c2:
                fig_rec = px.bar(rec_results, x='match_score', y='track_name', 
                                 orientation='h', color='match_score',
                                 title="Độ phù hợp (%)", color_continuous_scale='Agsunset')
                st.plotly_chart(fig_rec, use_container_width=True)

    # Xuất JSON cho Slide báo cáo
    with st.expander("📄 Dữ liệu JSON đầu ra (Cho API/Database)"):
        st.json(profiles_data)

else:
    st.info("Nhấn nút 'Chạy Hệ Thống' ở Sidebar để bắt đầu mô phỏng.")