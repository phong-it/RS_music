import sys
from collections import deque
import pandas as pd

class BehavioralProcessor:
    def __init__(self, max_store_size=10000):
        # Lưu trữ các bản ghi đã xử lý
        self.feature_store = deque(maxlen=max_store_size)
        # Quản lý trạng thái user (tích lũy)
        self.user_states = {}

    def _get_user_state(self, user_id):
        if user_id not in self.user_states:
            self.user_states[user_id] = {
                'total_plays': 0,
                'total_skips': 0,
                'listen_order': 0
            }
        return self.user_states[user_id]

    def process_real_time_event(self, event_data):
        """Xử lý logic hành vi và trả về feature record."""
        user_id = event_data['user_id']
        state = self._get_user_state(user_id)

        # 1. Cập nhật State
        state['listen_order'] += 1
        state['total_plays'] += 1

        # 2. Logic Skip
        is_skipped = event_data['play_duration_sec'] < 30
        if is_skipped:
            state['total_skips'] += 1

        current_skip_rate = state['total_skips'] / state['total_plays']

        # 3. Chuẩn hóa Search input
        raw_searched = event_data.get('is_searched', False)
        is_searched = str(raw_searched).strip().upper() == 'TRUE' if isinstance(raw_searched, str) else bool(raw_searched)

        # 4. Tính Engagement Score (Trọng tâm của Khôi)
        if is_skipped:
            engagement_score = 0.0
        else:
            base_score = (event_data['play_duration_sec'] * 0.5) + \
                         (event_data['frequency'] * 2.0) + \
                         (event_data['dwell_time_sec'] * 0.1)
            search_multiplier = 1.5 if is_searched else 1.0
            user_quality_penalty = 1.0 - (current_skip_rate * 0.2)
            engagement_score = round(base_score * search_multiplier * user_quality_penalty, 2)

        # 5. Tạo bản ghi hoàn chỉnh
        feature_record = {
            'user_id': user_id,
            'song_id': event_data['song_id'],
            'is_skipped': int(is_skipped), # Chuyển về 0/1 cho Phong dễ chạy ML
            'skip_rate': round(current_skip_rate, 2),
            'listen_order': state['listen_order'],
            'engagement_score': engagement_score,
            'is_searched': int(is_searched)
        }

        self.feature_store.append(feature_record)
        return feature_record

    def process_csv_stream(self, file_path):
        """Đọc file và trả về DataFrame để Phong (Người D) sử dụng."""
        try:
            df_raw = pd.read_csv(file_path)
            if df_raw.empty:
                return pd.DataFrame()

            # Xử lý từng dòng và thu thập kết quả
            processed_list = []
            for _, row in df_raw.iterrows():
                result = self.process_real_time_event(row.to_dict())
                processed_list.append(result)

            # Chuyển đổi toàn bộ sang DataFrame
            final_df = pd.DataFrame(processed_list)
            print(f"✅ Đã xử lý {len(final_df)} sự kiện. Dữ liệu đã sẵn sàng cho Phong!")
            return final_df

        except Exception as e:
            print(f"❌ LỖI: {e}")
            return pd.DataFrame()

# --- Ô CHẠY TRÊN COLAB ---
processor = BehavioralProcessor()
# Khôi lưu kết quả vào biến 'output_for_phong'
output_for_phong = processor.process_csv_stream('data_thuc_te.csv')

# Hiển thị thử kết quả của Khôi
output_for_phong.head()