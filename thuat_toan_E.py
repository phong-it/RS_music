from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import json
import asyncio
import time

# =====================================================================
# [GIAO KÈO VỚI NGƯỜI D]: 
# D phải gom code của họ vào file `service_d_model.py` 
# và tạo hàm `get_all_recommendations(user_id, context_data, blacklist)`
# Hàm này TRẢ VỀ MỘT DICTIONARY chứa cả list cá nhân hóa và trending.
# =====================================================================
try:
    from service_d_model import get_all_recommendations
except ImportError:
    # Fallback giả lập nếu D chưa nộp code
    def get_all_recommendations(user_id, context_data, blacklist):
        return {
            "personalized": [{"song_id": 42, "title": "Jazz in Rain", "score": 0.98, "reason": "DeepFM Predict"}],
            "trending": [
                {"song_id": 99, "title": "Top Hits Nhạc Trẻ", "views": 150000},
                {"song_id": 88, "title": "Viral TikTok", "views": 120000}
            ]
        }

app = FastAPI(title="Spotify Clone - Mixed Hybrid API Gateway")

class ClientRequest(BaseModel):
    timezone_offset: int = 7
    device: str = "Mobile"

# =====================================================================
# TẦNG 1: KẾT NỐI VẬT LÝ ĐẾN CÁC MICROSERVICES A VÀ C (Giữ nguyên)
# =====================================================================

async def fetch_A_profile(user_id: int):
    """ĐỌC FILE THẬT: Lấy Profile và Blacklist do Streamlit của A tạo ra"""
    try:
        def read_json():
            with open("user_audio_profiles.json", "r", encoding="utf-8") as f:
                return json.load(f)
        
        profiles = await asyncio.to_thread(read_json)
        target_user = f"user_{user_id:02d}"
        
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
    """GỌI API THẬT: Bắn HTTP Request sang Server FastAPI của C"""
    url = f"http://localhost:8001/analyze/{user_id}/1"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=1.5)
            if response.status_code == 200:
                data = response.json()
                return data.get("context_feature", {})
            return {}
        except Exception as e:
            print(f"[Cảnh báo] Server C đang sập hoặc timeout: {e}")
            return {"time_weight": 0.5, "avg_skip_rate": 0.5}

# =====================================================================
# GỌI NGƯỜI D (ĐÃ CẬP NHẬT KIẾN TRÚC MỚI)
# =====================================================================

async def fetch_D_recommendations(user_id: int, context_data: dict, profile_data: dict):
    """GỌI HÀM THẬT: Bơm data cho D để D trả về CẢ Cá nhân hóa & Trending"""
    try:
        blacklist = profile_data.get("blocked_artists", [])
        
        # Vẫn phải bọc trong to_thread vì D chạy thuật toán (CPU-bound)
        recs_dict = await asyncio.to_thread(
            get_all_recommendations, 
            user_id, context_data, blacklist
        )
        return recs_dict
    except Exception as e:
        print(f"[Lỗi Mô hình D]: {e}")
        # Trả về format chuẩn để App không bị lỗi
        return {"personalized": [], "trending": []}

# =====================================================================
# TẦNG 2: ORCHESTRATION (NHẠC TRƯỞNG ĐIỀU PHỐI E)
# =====================================================================

@app.post("/api/feed/{user_id}")
async def get_home_feed(user_id: int, client_req: ClientRequest):
    start_time = time.time()
    
    # --- BƯỚC 1: HỎI A VÀ C CÙNG LÚC (SONG SONG) ---
    task_a = asyncio.wait_for(fetch_A_profile(user_id), timeout=2.0)
    task_c = asyncio.wait_for(fetch_C_context(user_id), timeout=2.0)
    
    step1_results = await asyncio.gather(task_a, task_c, return_exceptions=True)
    
    user_profile = step1_results[0] if not isinstance(step1_results[0], Exception) else {}
    user_context = step1_results[1] if not isinstance(step1_results[1], Exception) else {}

    # --- BƯỚC 2: GIAO TOÀN BỘ TRÁCH NHIỆM XẾP HẠNG CHO D ---
    try:
        # Ép D phải trả kết quả trong 3.5s (cho thêm 0.5s vì D làm 2 việc)
        d_results = await asyncio.wait_for(
            fetch_D_recommendations(user_id, user_context, user_profile), 
            timeout=3.5
        )
    except asyncio.TimeoutError:
        print("[Cảnh báo] D chạy quá 3.5s, kích hoạt Fallback.")
        d_results = {"personalized": [], "trending": []}
    except Exception as e:
        print(f"[Lỗi D khi điều phối]: {e}")
        d_results = {"personalized": [], "trending": []}

    # Bóc tách kết quả D trả về
    personalized_recs = d_results.get("personalized", [])
    trending_recs = d_results.get("trending", [])

    # --- BƯỚC 3: ĐÓNG GÓI MIXED HYBRID JSON (Trách nhiệm của E) ---
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

    # Dựa vào data của D, E tiến hành chia Buckets hiển thị UI
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