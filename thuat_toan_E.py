from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import json
import asyncio
import time

# =====================================================================
# [GIAO KÈO VỚI NGƯỜI D]: 
# D phải gom code của họ vào file `service_d_model.py` 
# và tạo hàm `get_top_5_recommendations(user_id, context_data, blacklist)`
# =====================================================================
try:
    from service_d_model import get_top_5_recommendations
except ImportError:
    # Fallback giả lập nếu D chưa nộp code
    def get_top_5_recommendations(user_id, context_data, blacklist):
        return [{"song_id": 42, "title": "Jazz in Rain", "score": 0.98, "reason": "DeepFM Predict"}]

app = FastAPI(title="Spotify Clone - Mixed Hybrid API Gateway")

class ClientRequest(BaseModel):
    timezone_offset: int = 7
    device: str = "Mobile"

# =====================================================================
# TẦNG 1: KẾT NỐI VẬT LÝ ĐẾN CÁC MICROSERVICES
# =====================================================================

async def fetch_A_profile(user_id: int):
    """ĐỌC FILE THẬT: Lấy Profile và Blacklist do Streamlit của A tạo ra"""
    try:
        # Dùng asyncio.to_thread để việc đọc file không làm đứng Server
        def read_json():
            with open("user_audio_profiles.json", "r", encoding="utf-8") as f:
                return json.load(f)
        
        profiles = await asyncio.to_thread(read_json)
        target_user = f"user_{user_id:02d}" # Format "user_01" của A
        
        for p in profiles:
            if p["user_id"] == target_user:
                return p
        return {"blocked_artists": [], "favorite_artists": []}
    except FileNotFoundError:
        print("[Cảnh báo] Không tìm thấy file JSON của A.")
        return {"blocked_artists": [], "favorite_artists": []}
    except Exception as e:
        print(f"[Lỗi A]: {e}")
        return {"blocked_artists": [], "favorite_artists": []}

async def fetch_C_context(user_id: int):
    """GỌI API THẬT: Bắn HTTP Request sang Server FastAPI của C (Cổng 8001)"""
    # Mẹo: Gọi tạm song_id = 1 để C trả về Context của User
    url = f"http://localhost:8001/analyze/{user_id}/1"
    
    # Dùng httpx (chuẩn Async) thay cho requests (chạy đồng bộ, gây lag)
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=1.5) # Chỉ đợi C tối đa 1.5s
            if response.status_code == 200:
                data = response.json()
                return data.get("context_feature", {})
            return {}
        except Exception as e:
            print(f"[Cảnh báo] Server C đang sập hoặc timeout: {e}")
            # Fallback an toàn nếu C sập
            return {"time_weight": 0.5, "avg_skip_rate": 0.5}

async def fetch_D_deepfm(user_id: int, context_data: dict, profile_data: dict):
    """GỌI HÀM THẬT: Chạy Model TensorFlow của D"""
    try:
        # TỐI QUAN TRỌNG: Model AI chạy tính toán bằng CPU/GPU rất nặng.
        # Phải ném nó vào to_thread để nó chạy ngầm, nếu không toàn bộ hệ thống API sẽ bị treo!
        blacklist = profile_data.get("blocked_artists", [])
        recs = await asyncio.to_thread(
            get_top_5_recommendations, 
            user_id, context_data, blacklist
        )
        return recs
    except Exception as e:
        print(f"[Lỗi Mô hình D]: {e}")
        return []

async def fetch_Trending():
    """Giả lập lấy danh sách Trending (Vì A chưa cấp API Trending)"""
    await asyncio.sleep(0.05)
    return [
        {"song_id": 99, "title": "Top Hits Nhạc Trẻ", "views": 150000},
        {"song_id": 88, "title": "Viral TikTok", "views": 120000}
    ]

# =====================================================================
# TẦNG 2: ORCHESTRATION (NHẠC TRƯỞNG ĐIỀU PHỐI)
# =====================================================================

@app.post("/api/feed/{user_id}")
async def get_home_feed(user_id: int, client_req: ClientRequest):
    start_time = time.time()
    
    # --- BƯỚC 1: HỎI A VÀ C CÙNG LÚC (SONG SONG) ---
    task_a = asyncio.wait_for(fetch_A_profile(user_id), timeout=2.0)
    task_c = asyncio.wait_for(fetch_C_context(user_id), timeout=2.0)
    
    # Chạy và gom kết quả. Nếu lỗi, return_exceptions=True giúp App không bị sập.
    step1_results = await asyncio.gather(task_a, task_c, return_exceptions=True)
    
    user_profile = step1_results[0] if not isinstance(step1_results[0], Exception) else {}
    user_context = step1_results[1] if not isinstance(step1_results[1], Exception) else {}

    # --- BƯỚC 2: BƠM DATA CHO D CHẠY, ĐỒNG THỜI LẤY TRENDING ---
    task_d = asyncio.wait_for(fetch_D_deepfm(user_id, user_context, user_profile), timeout=3.0)
    task_trend = asyncio.wait_for(fetch_Trending(), timeout=1.0)
    
    step2_results = await asyncio.gather(task_d, task_trend, return_exceptions=True)
    
    personalized_recs = step2_results[0] if not isinstance(step2_results[0], Exception) else []
    trending_recs = step2_results[1] if not isinstance(step2_results[1], Exception) else []

    # --- BƯỚC 3: ĐÓNG GÓI MIXED HYBRID JSON ---
    time_weight = user_context.get("time_weight", 0.5)
    mood = "Buổi tối thư giãn" if time_weight < 0.8 else "Ngày mới năng động"

    response = {
        "status": "success",
        "metadata": {
            "user_id": user_id,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "applied_blacklist": user_profile.get("blocked_artists", [])
        },
        "greeting": f"Sẵn sàng cho {mood} chưa?",
        "feed": []
    }

    if personalized_recs:
        response["feed"].append({
            "section": "cascade_deepfm",
            "title": "Dành riêng cho bạn (Gợi ý AI)",
            "data": personalized_recs
        })
        
    if trending_recs:
        response["feed"].append({
            "section": "trending",
            "title": "Cả thế giới đang nghe",
            "data": trending_recs
        })

    return response