import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

class BehavioralProcessor:
    def __init__(self):
        self.feature_store_mock = [] # dùng list lưu trữ luồng sự kiện liên tục
    
    def create_mock_data(self):
        data = {
            'user_id': ['U1', 'U1', 'U2', 'U3'],
            'song_id': ['S101', 'S102', 'S101', 'S105'],
            'play_duration_sec': [120, 15, 200, 45], # Thời gian nghe 
            'frequency': [3, 1, 5, 2],               # Tần suất lặp lại 
            'dwell_time_sec': [10, 2, 30, 5],         # Dwell time trên màn hình    
            'is_searched': [True, False, False, True]   # Lịch sử tìm kiếm
        }
        return pd.DataFrame(data)
    
    def process_real_time_event(self, event_data):
        is_skipped = event_data['play_duration_sec'] < 30

        if is_skipped:
            engagement_score = 0.0
        else:
            base_score = (event_data['play_duration_sec'] * 0.5) + \
                         (event_data['frequency']* 2.0) + \
                         (event_data['dwell_time_sec'] * 0.1)
            
            search_multiplier = 1.5 if event_data.get('is_searched', False) else 1.0

            engagement_score = round(base_score * search_multiplier, 2)

        feature_record = {
            'user_id': event_data['user_id'],
            'song_id': event_data['song_id'],
            'is_skipped': is_skipped,
            'engagement_score': engagement_score
        }

        self.feature_store_mock.append(feature_record)
        return feature_record
    
    def process_batch_vectorized(self, raw_df):
        raw_df['is_skipped'] = raw_df['play_duration_sec'] <30

        raw_df['base_score'] = (raw_df['play_duration_sec'] * 0.5) + \
                               (raw_df['frequency'] * 2.0) + \
                               (raw_df['dwell_time_sec'] * 0.1)
        
        raw_df['search_multiplier'] = np.where(raw_df['is_searched'] == True, 1.5, 1.0)

        raw_df['engagement_score'] = raw_df['base_score'] * raw_df['search_multiplier']

        raw_df.loc[raw_df['is_skipped'] == True, 'engagement_score'] = 0.0

        raw_df['engagement_score'] = raw_df['engagement_score'].round(2)

        return raw_df

if __name__ == "__main__":
    processor = BehavioralProcessor()

    print("Kiểm tra luồng tức thời: ")
    new_event = {
        'user_id': 'U99', 'song_id': 'S999', 
        'play_duration_sec': 150, 'frequency': 1, 
        'dwell_time_sec': 15, 'is_searched': True
    }      
    result = processor.process_real_time_event(new_event)
    print(f"Sự kiện vừa nảy số: {result}\n")

    print("Kiểm tra hiệu năng: ")
    raw_data = processor.create_mock_data()
    optimized_data = processor.process_batch_vectorized(raw_data)
    print(optimized_data[['user_id', 'song_id', 'is_skipped', 'is_searched', 'engagement_score']])
