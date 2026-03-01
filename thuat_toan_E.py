from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import time

app = FastAPI(title="Music AI Gateway (Mixed Hybrid Orchestrator - Pro Version)")

# =====================================================================
# 1. ĐỊNH NGHĨA REQUEST TỪ CLIENT (MOBILE APP)
# =====================================================================
class ClientTelemetry(BaseModel):
    client_timezone_offset: int  # VD: +7 cho Việt Nam (Để fix lỗi múi giờ của C)
    device_type: str             # "Mobile", "Desktop"

# =====================================================================
# 2. GIẢ LẬP GỌI CÁC MICROSERVICES (Thực tế dùng thư viện httpx)
# =====================================================================

async def fetch_service_a_profile(user_id: int):
    """
    GỌI NGƯỜI A: Lấy file JSON Hồ sơ người dùng (Từ code Streamlit).
    """
    await asyncio.sleep(0.05) # Giả lập độ trễ mạng
    return {
        "favorite_artists": ["Pop", "EDM"],
        "blocked_artists": ["Rock"], # Dùng để D loại trừ
        "audio_fingerprint": {"tempo": 120, "energy": 0.8}
    }

async def fetch_service_c_context(user_id: int):
    """
    GỌI NGƯỜI C: Lấy Context (time_weight, avg_skip_rate) của user.
    (Giả định C cung cấp thêm API /context/{user_id} bên cạnh API /analyze)
    """
    await asyncio.sleep(0.03)
    # Lấy output chuẩn từ code mới của Người C
    return {
        "time_weight": 0.7,      # Output từ Tầng 1B của C
        "avg_skip_rate": 0.15    # Dữ liệu Numerical cực quan trọng cho DeepFM của D
    }

async def fetch_service_d_deepfm(user_id: int, profile: dict, context: dict):
    """
    GỌI NGƯỜI D: Gửi Profile (A) và Context (C) sang cho D chạy DeepFM Pro.
    D sẽ dùng `avg_skip_rate` làm Numerical Input, `favorite_artists` làm Categorical Input.
    """
    await asyncio.sleep(0.2) # D chạy Two-Tower + DeepFM khá nặng (200ms)
    return [
        {"song_id": 42, "genre": "Pop", "match_score": 0.98, "reason": "Hợp gu + Ít skip"},
        {"song_id": 108, "genre": "EDM", "match_score": 0.94, "reason": "Hợp bối cảnh tối"},
        {"song_id": 3, "genre": "Lofi", "match_score": 0.88, "reason": "Khám phá mới"}
    ]

async def fetch_service_a_trending():
    """GỌI NGƯỜI A: Lấy danh sách Top Trending."""
    await asyncio.sleep(0.02)
    return [
        {"song_id": 999, "title": "Top Viral TikTok", "views": 5000000},
        {"song_id": 888, "title": "Lofi Chill", "views": 3500000}
    ]

# =====================================================================
# 3. API GATEWAY CHÍNH - NHẠC TRƯỞNG E
# =====================================================================

@app.post("/api/v2/home_feed/{user_id}")
async def get_home_feed(user_id: int, telemetry: ClientTelemetry):
    start_time = time.time()
    
    # -----------------------------------------------------------------
    # BƯỚC 1: LẤY DỮ LIỆU ĐẦU VÀO (A và C) SONG SONG
    # -----------------------------------------------------------------
    try:
        task_a = asyncio.wait_for(fetch_service_a_profile(user_id), timeout=0.1)
        task_c = asyncio.wait_for(fetch_service_c_context(user_id), timeout=0.1)
        
        results_step_1 = await asyncio.gather(task_a, task_c, return_exceptions=True)
        
        user_profile = results_step_1[0] if not isinstance(results_step_1[0], Exception) else {"blocked_artists": []}
        user_context = results_step_1[1] if not isinstance(results_step_1[1], Exception) else {"avg_skip_rate": 0.5, "time_weight": 0.5}
        
    except Exception as e:
        print(f"[Cảnh báo] Lỗi Bước 1: {e}")
        user_profile, user_context = {"blocked_artists": []}, {"avg_skip_rate": 0.5, "time_weight": 0.5}

    # -----------------------------------------------------------------
    # BƯỚC 2: GỌI RANKING (D) VÀ TRENDING (A) SONG SONG
    # -----------------------------------------------------------------
    try:
        # Bơm dữ liệu từ B1 vào cho D
        task_d = asyncio.wait_for(fetch_service_d_deepfm(user_id, user_profile, user_context), timeout=0.3)
        task_trend = asyncio.wait_for(fetch_service_a_trending(), timeout=0.1)
        
        results_step_2 = await asyncio.gather(task_d, task_trend, return_exceptions=True)
        
        deepfm_recs = results_step_2[0] if not isinstance(results_step_2[0], Exception) else []
        trending_recs = results_step_2[1] if not isinstance(results_step_2[1], Exception) else []
        
    except Exception as e:
        print(f"[Cảnh báo] Lỗi Bước 2: {e}")
        deepfm_recs, trending_recs = [], []

    # -----------------------------------------------------------------
    # BƯỚC 3: ĐÓNG GÓI JSON (MIXED HYBRID) VÀ PHẢN HỒI
    # -----------------------------------------------------------------
    response = {
        "user_id": user_id,
        "diagnostics": {
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "user_skip_rate": user_context.get("avg_skip_rate")
        },
        "feed_sections": []
    }

    # Section 1: Cá nhân hóa (Từ D)
    if deepfm_recs:
        response["feed_sections"].append({
            "type": "deepfm_cascade",
            "title": "Gợi ý thông minh cho bạn",
            "items": deepfm_recs
        })
        
    # Section 2: Trending (Từ A)
    if trending_recs:
        response["feed_sections"].append({
            "type": "trending",
            "title": "Đang thịnh hành",
            "items": trending_recs
        })

    return response