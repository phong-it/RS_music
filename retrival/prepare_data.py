import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder


class SmartDataProcessor:
    def __init__(self):
        self.user_map = {}
        self.song_map = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        # THÊM VÀO ĐÂY: Khởi tạo các kho chứa để hàm test có thể truy cập
        self.user_features_dict = {}
        self.song_features_dict = {}
        self.context_features_dict = {}

    def _normalize_id(self, raw_id):
        """Đưa ID về chuẩn: ví dụ 'User_01', 'user_01', 1 -> 'user_01'"""
        if str(raw_id).isdigit():
            return f"user_{int(raw_id):02d}"
        return str(raw_id).strip().lower()

    def _dynamic_get_index(self, entity_id, mapping):
        if entity_id not in mapping:
            mapping[entity_id] = len(mapping)
        return mapping[entity_id]

    def process_all(self, path_a, df_b, path_c):
        # Đọc dữ liệu
        with open(path_a, 'r', encoding='utf-8') as f:
            data_a = json.load(f)
        with open(path_c, 'r', encoding='utf-8') as f:
            data_c = json.load(f)
        df_b = df_b.copy()

        # --- BƯỚC 1: XÂY DỰNG USER FEATURES (Dùng self. để lưu lại) ---
        self.user_features_dict = {}
        for profile in data_a:
            u_id = self._normalize_id(profile['user_id'])
            u_idx = self._dynamic_get_index(u_id, self.user_map)

            audio_feats = list(profile.get('audio_fingerprint', {}).values())
            top_tags = profile['metadata']['top_tags'][0] if profile['metadata']['top_tags'] else "Unknown"

            # Lưu vào thuộc tính lớp
            self.user_features_dict[u_idx] = {
                'num': audio_feats, 'cat': [top_tags]}

        # --- BƯỚC 2: XÂY DỰNG SONG & CONTEXT (Dùng self. để lưu lại) ---
        self.song_features_dict = {}
        self.context_features_dict = {}

        for entry in data_c:
            s_id = self._normalize_id(entry['content_analysis']['song_id'])
            u_id = self._normalize_id(entry['context_analysis']['user_id'])

            s_idx = self._dynamic_get_index(s_id, self.song_map)
            u_idx = self._dynamic_get_index(u_id, self.user_map)

            s_info = entry['content_analysis']
            self.song_features_dict[s_idx] = {
                'num': [s_info['bpm']],
                'cat': [s_info['mood_detected'], s_info['genre']]
            }
            self.context_features_dict[(u_idx, s_idx)] = [
                entry['context_analysis']['device'],
                entry['context_analysis']['inferred_vibe']
            ]

        # --- BƯỚC 3: GHÉP NỐI VỚI FILE B (KHÔI) ---
        final_rows_cat, final_rows_num, final_y = [], [], []

        for _, row in df_b.iterrows():
            norm_u_id = self._normalize_id(row['user_id'])
            norm_s_id = self._normalize_id(row['song_id'])

            u_idx = self.user_map.get(norm_u_id)
            s_idx = self.song_map.get(norm_s_id)

            # Kiểm tra khớp ID để tránh rỗng dữ liệu
            if u_idx is not None and s_idx is not None:
                cat_feat = (self.user_features_dict[u_idx]['cat'] +
                            self.song_features_dict[s_idx]['cat'] +
                            self.context_features_dict.get((u_idx, s_idx), ["Unknown", "Chill"]))

                # Tính toán skip_rate nếu cột tồn tại, nếu không mặc định 0.0
                skip_rate = float(row.get('skip_rate', 0.0))
                num_feat = (self.user_features_dict[u_idx]['num'] +
                            self.song_features_dict[s_idx]['num'] +
                            [skip_rate])

                final_rows_cat.append(cat_feat)
                final_rows_num.append(num_feat)
                final_y.append(float(row.get('engagement_score', 0.0)))

        if len(final_rows_cat) == 0:
            print("LỖI: Vẫn không khớp ID. Hãy kiểm tra định dạng ID trong file CSV!")
            return None, None, None

        # --- BƯỚC 4: ENCODING & SCALING ---
        X_cat = np.array(final_rows_cat)
        for i in range(X_cat.shape[1]):
            le = LabelEncoder()
            X_cat[:, i] = le.fit_transform(X_cat[:, i].astype(str))
            self.encoders[f'col_{i}'] = le

        X_num = self.scaler.fit_transform(np.array(final_rows_num))
        y = np.array(final_y)

        print(f"Đã khớp {len(final_rows_cat)} dòng dữ liệu.")
        return X_cat.astype(int), X_num, y

# --- CẬP NHẬT HÀM TEST ĐỂ KHỚP VỚI PROCESSOR ---


def get_recommendations_for_user(target_user):
    try:
        # Sử dụng bộ lọc ID chuẩn hóa
        norm_id = processor._normalize_id(target_user)
        u_idx = processor.user_map.get(norm_id)

        if u_idx is None:
            print(f"User {target_user} không tồn tại!")
            return

        # Bây giờ processor.user_features_dict đã tồn tại nhờ dùng self.
        user_raw_cat = processor.user_features_dict[u_idx]['cat']
        user_raw_num = processor.user_features_dict[u_idx]['num']

        print(f"Đã lấy đặc trưng cho {norm_id}: {user_raw_cat}")
        # (Tiếp tục logic DeepFM của bạn ở đây...)

    except Exception as e:
        print(f"Lỗi khi test: {e}")
