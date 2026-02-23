import pandas as pd;
import numpy as np;

class BehavioralProcessor:
    def __init__(self):
        self.feature_store_mock = {}
    
    def create_mock_data(self):
        data = {
            'user_id': ['U1', 'U1', 'U2', 'U3'],
            'song_id': ['S101', 'S102', 'S101', 'S105'],
            'play_duration_sec': [120, 15, 200, 45], # Thời gian nghe [cite: 49]
            'frequency': [3, 1, 5, 2],               # Tần suất lặp lại [cite: 49]
            'dwell_time_sec': [10, 2, 30, 5]         # Dwell time trên màn hình [cite: 49]    
        }
        return pd.DataFrame(data)
    
    def is_skipped(self, duration_sec):
        if duration_sec < 30:
            return True
        else:
            return False
    
    def calculate_engagement_score(self, row):
        if row['is_skipped'] == True:
            return 0.0
        else:
            duration = row['play_duration_sec']
            freq = row['frequency']
            dwell = row['dwell_time_sec']

            score = (duration * 0.5) + (freq * 2.0) + (dwell * 0.1)

            return round(score, 2) 
    
    def process_stream(self, raw_df):
        print("Đang xử lý luồng dữ liệu hành vi...")

        raw_df['is_skipped'] = raw_df['play_duration_sec'].apply(self.is_skipped)

        raw_df['engagement_score'] = raw_df.apply(self.calculate_engagement_score, axis=1)

        return raw_df

    def push_to_feature_store(self, processed_df):
        final_features = processed_df[['user_id', 'song_id', 'engagement_score', 'is_skipped']]

        self.feature_store_mock = final_features.to_dict(orient='records')

        print("Đã đẩy dữ liệu đặc trưng vào Feature Store thành công!")
        print("Nội dung trong kho hiện tại:")
        print(self.feature_store_mock)

if __name__ == "__main__":
    processor = BehavioralProcessor()

    raw_data = processor.create_mock_data()
    print("Dữ liệu thô ban đầu:")
    print(raw_data)  
    print("\n")

    processed_data = processor.process_stream(raw_data)

    print("DỮ LIỆU SAU KHI XỬ LÝ: ")
    print(processed_data[['user_id', 'song_id', 'play_duration_sec', 'is_skipped', 'engagement_score']])
    print("\n")

    print("Kho lưu trữ")
    processor.push_to_feature_store(processed_data)      

