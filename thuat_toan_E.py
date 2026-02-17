from flask import Flask, jsonify, request
import time


# Service của Người D (Thuật toán Cascade)
def call_service_D_recommendation(user_id, context):
    return [
        {"id": 1, "title": "Bài Hát Đúng Gu 1", "score": 0.98},
        {"id": 2, "title": "Bài Hát Đúng Gu 2", "score": 0.95},
        {"id": 3, "title": "Bài Hát Đúng Gu 3", "score": 0.91}
    ]

# Service của Người A (Dữ liệu tĩnh/Trending)
def call_service_A_trending():
    return [
        {"id": 101, "title": "Hot Hit Tháng 10", "views": 50000},
        {"id": 102, "title": "Top Viral TikTok", "views": 48000}
    ]

# Service của Người C (Ngữ cảnh)
def call_service_C_get_context(user_ip, device_info):
    return {"time_of_day": "Evening", "activity": "Relaxing"}

# NGƯỜI E (INTEGRATION MANAGER) 

app = Flask(__name__)

@app.route('/api/home_feed', methods=['GET'])
def get_home_feed():
    """
    API chính trả về toàn bộ nội dung trang chủ.
    Đây là nơi E thực hiện Mixed Hybrid.
    """
    user_id = request.args.get('user_id')
    user_ip = request.remote_addr
    
    # 1. Lấy ngữ cảnh từ Người C (Để gửi cho D)
    try:
        current_context = call_service_C_get_context(user_ip, request.headers.get('User-Agent'))
    except Exception as e:
        # Fallback nếu C lỗi: Mặc định
        current_context = {"time_of_day": "Day", "activity": "Unknown"}

    final_response = {
        "greeting": f"Good {current_context['time_of_day']}!",
        "sections": [] # Các "Buckets" hiển thị 
    }

    # 2. BUCKET 1: "Dành riêng cho bạn" (Gọi Người D)
    #  Đảm bảo chịu tải: Dùng try-except để bắt lỗi nếu D sập
    try:
        # Giả sử đặt timeout là 200ms, nếu D tính lâu quá thì bỏ qua
        start_time = time.time()
        
        personal_recs = call_service_D_recommendation(user_id, current_context)
        
        # Nếu D trả về kết quả rỗng (Cold start), E tự quyết định không hiện section này
        if personal_recs:
            final_response["sections"].append({
                "type": "personal_cascade",
                "title": "Gợi ý riêng cho bạn",
                "data": personal_recs
            })
            
    except Exception as e:
        print(f"Service D bị lỗi hoặc timeout: {e}")
        # Không làm sập app, chỉ đơn giản là không hiện section này
        pass

    # 3. BUCKET 2: "Xu hướng" (Gọi Người A)
    # Đây là phần "Mixed" - Trộn lẫn nội dung cá nhân và đại chúng
    try:
        trending_list = call_service_A_trending()
        final_response["sections"].append({
            "type": "trending",
            "title": "Đang thịnh hành",
            "data": trending_list
        })
    except Exception:
        # Nếu cả A cũng lỗi thì trả về list rỗng
        pass

    #  Phản hồi kết quả cuối cùng cho User
    return jsonify(final_response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)