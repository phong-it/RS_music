from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import time

app = FastAPI(title="Spotify-like API Gateway (Mixed Hybrid Orchestrator)")

# =====================================================================
# 1. ĐỊNH NGHĨA DỮ LIỆU TỪ CLIENT GỬI LÊN (MOBILE APP)
# =====================================================================
class ClientTelemetry(BaseModel):
    client_hour: int      # Giờ tại máy người dùng (0-23)
    device: str           # "headphones", "speaker"
    is_moving: bool       # True/False

# =====================================================================
# 2. GIẢ LẬP GỌI CÁC MICROSERVICES (Dùng httpx trong thực tế)
# =====================================================================

async def fetch_service_a_profile(user_id: str):
    """
    GỌI NGƯỜI A: Lấy Hồ sơ tĩnh của User (Audio Fingerprint & Blacklist).
    Giả lập đọc từ Redis hoặc file JSON mà code Streamlit của Người A đã sinh ra.
    """
    await asyncio.sleep(0.02) # Cache Redis cực nhanh (20ms)
    return {
        "favorite_artists": ["Sơn Tùng M-TP", "Binz"],
        "blocked_artists": ["Jack"], # Danh sách cực kỳ quan trọng để gửi cho D
        "audio_fingerprint": {"energy": 0.8, "tempo": 120}
    }

async def fetch_service_c_context(user_id: str, telemetry: ClientTelemetry):
    """
    GỌI NGƯỜI C: Gửi Telemetry để lấy Context chuẩn hóa.
    """
    await asyncio.sleep(0.03) # 30ms
    time_context = "evening" if 18 <= telemetry.client_hour <= 23 else "day"
    return {
        "time_context": time_context,
        "preferred_mood": "calm" if time_context == "evening" else "energetic",
        "device": telemetry.device
    }

async def fetch_service_d_recommender(user_id: str, profile: dict, context: dict):
    """
    GỌI NGƯỜI D: Chạy DeepFM. 
    Lưu ý: E phải bơm profile (có Blacklist của A) và context (của C) vào cho D.
    """
    await asyncio.sleep(0.15) # D tính toán nặng (150ms)
    # Giả sử D đã lấy blacklist ra để loại bỏ các bài hát của "Jack"
    return [
        {"id": "s1", "title": "Cơn Mưa Ngang Qua", "artist": "Sơn Tùng M-TP", "match_score": 0.98},
        {"id": "s2", "title": "Bigcityboi", "artist": "Binz", "match_score": 0.95}
    ]

async def fetch_service_a_trending():
    """GỌI NGƯỜI A: Lấy danh sách Top Trending (Bucket 2)."""
    await asyncio.sleep(0.02) # 20ms
    return [
        {"id": "t1", "title": "Nhạc Hot TikTok 2026", "views": 5000000},
        {"id": "t2", "title": "Lofi Chill Quán Cà Phê", "views": 3500000}
    ]

# =====================================================================
# 3. API GATEWAY CHÍNH - LUỒNG ĐI PHỨC TẠP NHƯNG SIÊU NHANH
# =====================================================================

@app.post("/api/v1/home_feed/{user_id}")
async def get_home_feed(user_id: str, telemetry: ClientTelemetry):
    start_time = time.time()
    
    # -----------------------------------------------------------------
    # BƯỚC 1: Lấy thông tin đầu vào (A và C) SONG SONG
    # D cần Profile của A và Context của C để chạy. Nên ta gọi A và C cùng lúc.
    # -----------------------------------------------------------------
    try:
        task_a_profile = asyncio.wait_for(fetch_service_a_profile(user_id), timeout=0.1)
        task_c_context = asyncio.wait_for(fetch_service_c_context(user_id, telemetry), timeout=0.1)
        
        results_step_1 = await asyncio.gather(task_a_profile, task_c_context, return_exceptions=True)
        
        user_profile = results_step_1[0] if not isinstance(results_step_1[0], Exception) else {"blocked_artists": []}
        user_context = results_step_1[1] if not isinstance(results_step_1[1], Exception) else {"preferred_mood": "neutral"}
        
    except Exception as e:
        print(f"Lỗi cục bộ ở Bước 1: {e}")
        user_profile, user_context = {"blocked_artists": []}, {"preferred_mood": "neutral"}

    # -----------------------------------------------------------------
    # BƯỚC 2: Gọi Thuật toán D (Cá nhân) và A (Trending) SONG SONG
    # -----------------------------------------------------------------
    try:
        # Bơm dữ liệu từ B1 vào cho D
        task_d = asyncio.wait_for(fetch_service_d_recommender(user_id, user_profile, user_context), timeout=0.25)
        task_a_trend = asyncio.wait_for(fetch_service_a_trending(), timeout=0.1)
        
        results_step_2 = await asyncio.gather(task_d, task_a_trend, return_exceptions=True)
        
        personal_recs = results_step_2[0] if not isinstance(results_step_2[0], Exception) else []
        trending_recs = results_step_2[1] if not isinstance(results_step_2[1], Exception) else []
        
    except Exception as e:
        print(f"Lỗi cục bộ ở Bước 2: {e}")
        personal_recs, trending_recs = [], []

    # -----------------------------------------------------------------
    # BƯỚC 3: Đóng gói JSON trả về App
    # -----------------------------------------------------------------
    mood = user_context.get('preferred_mood', 'chill')
    
    response = {
        "user_id": user_id,
        "greeting": f"Sẵn sàng cho âm nhạc {mood} chưa?",
        "diagnostics": {
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "applied_blacklist_count": len(user_profile.get("blocked_artists", []))
        },
        "feed_sections": []
    }

    # BUCKET 1: Cascade (Cá nhân hóa cao độ)
    if personal_recs:
        response["feed_sections"].append({
            "type": "personalized",
            "title": f"Gợi ý hoàn hảo cho bạn lúc này",
            "items": personal_recs
        })
        
    # BUCKET 2: Mixed (Xu hướng)
    if trending_recs:
        response["feed_sections"].append({
            "type": "trending",
            "title": "Cả thế giới đang nghe gì?",
            "items": trending_recs
        })

    return response