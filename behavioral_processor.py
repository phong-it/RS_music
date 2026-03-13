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
    
    def start_listening_stream(self):
        """
        Giả lập server API nhận dữ liệu liên tục từ giao diện người dùng (Front-end).
        Hệ thống sẽ chạy liên tục cho đến khi bạn ra lệnh tắt.
        """
        print("\n" + "="*55)
        print("🚀 HỆ THỐNG LẮNG NGHE SỰ KIỆN REAL-TIME ĐÃ KHỞI ĐỘNG")
        print("Gõ 'q' hoặc 'quit' ở phần User ID để tắt server.")
        print("="*55 + "\n")

        while True:
            try:
                # 1. Thu thập dữ liệu từ bàn phím (Giả lập luồng Real-time)
                user_id = input("👤 Nhập User ID (VD: U1): ")
                if user_id.lower() in ['q', 'quit', 'exit']:
                    print("🛑 Đang tắt hệ thống lắng nghe...")
                    break
                
                song_id = input("🎵 Nhập Song ID (VD: S101): ")
                duration = float(input("⏱️ Thời gian nghe (giây): "))
                freq = int(input("🔁 Số lần lặp lại: "))
                dwell = float(input("👀 Thời gian dừng xem màn hình (giây): "))
                
                search_input = input("🔍 Khách có chủ động tìm kiếm bài này không? (y/n): ").strip().lower()
                is_searched = True if search_input == 'y' else False

                # 2. Đóng gói dữ liệu thành chuẩn Event Dictionary
                live_event = {
                    'user_id': user_id,
                    'song_id': song_id,
                    'play_duration_sec': duration,
                    'frequency': freq,
                    'dwell_time_sec': dwell,
                    'is_searched': is_searched
                }

                # 3. Chuyền dữ liệu cho "Bộ não" tính điểm tức thời
                print("\n⏳ Hệ thống đang nảy số...")
                result = self.process_real_time_event(live_event)
                
                # 4. Trả kết quả lên màn hình
                print("✅ KẾT QUẢ TÍNH TOÁN:")
                print(f"   Khách hàng {result['user_id']} -> Bài hát {result['song_id']}")
                print(f"   Trạng thái Bỏ qua (Skip): {result['is_skipped']}")
                print(f"   Điểm Tương tác (Engagement): {result['engagement_score']}")
                
                # In xem trong Kho lưu trữ (Feature Store) đang có bao nhiêu bản ghi
                print(f"   [Kho Feature Store hiện đang lưu {len(self.feature_store_mock)} bản ghi]")
                print("-" * 55 + "\n")

            except ValueError:
                # Bắt lỗi nếu bạn lỡ gõ chữ vào chỗ yêu cầu nhập số
                print("❌ LỖI: Vui lòng nhập đúng định dạng số cho thời gian và tần suất!\n")

    def process_csv_stream(self, file_path):
        """
        Đọc dữ liệu từ file CSV và giả lập luồng sự kiện Real-time.
        Mỗi dòng trong file CSV sẽ được "bơm" vào hệ thống như một sự kiện độc lập.
        """
        print(f"\n📥 Đang nạp dữ liệu từ file: {file_path}...")
        try:
            # Dùng Pandas để đọc file CSV
            df = pd.read_csv(file_path)
            
            # Kiểm tra xem file có bị trống không
            if df.empty:
                print("❌ File CSV không có dữ liệu!")
                return
                
            print(f"✅ Đã tìm thấy {len(df)} sự kiện. Bắt đầu nảy số theo luồng...\n")
            
            # Chuyển đổi bảng dữ liệu thành danh sách các sự kiện (Dictionary)
            events = df.to_dict(orient='records')
            
            # Bơm từng sự kiện vào hàm xử lý Real-time
            for index, live_event in enumerate(events):
                result = self.process_real_time_event(live_event)
                
                # In tiến trình ra màn hình để theo dõi
                print(f"[{index + 1}/{len(df)}] Khách {result['user_id']} -> Bài {result['song_id']} | Bỏ qua: {result['is_skipped']} | Điểm: {result['engagement_score']}")
                
            print(f"\n🎉 Nạp file hoàn tất! Kho Feature Store hiện đang lưu tổng cộng {len(self.feature_store_mock)} bản ghi.")
            
        except FileNotFoundError:
            print(f"❌ LỖI: Không tìm thấy file '{file_path}'. Hãy đảm bảo file nằm cùng thư mục với code.")
        except Exception as e:
            print(f"❌ LỖI ĐỌC FILE: {e}")

# --- Phần chạy ứng dụng ---
if __name__ == "__main__":
    processor = BehavioralProcessor()
    
    # Bạn có thể mở khóa (bỏ dấu #) dòng dưới nếu muốn nhập tay như cũ
    # processor.start_listening_stream()
    
    # Chạy luồng dữ liệu tự động từ file CSV
    # Đảm bảo bạn đã tạo một file tên là 'data_thuc_te.csv' nằm cùng thư mục
    processor.process_csv_stream('data_thuc_te.csv')
