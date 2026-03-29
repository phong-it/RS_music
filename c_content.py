import pandas as pd
import json
from datetime import datetime


def professional_analysis_exporter(song_csv, user_csv, output_file="demo_analysis.json"):
    # 1. LOAD DỮ LIỆU THẬT TỪ ẢNH
    try:
        df_songs = pd.read_csv(song_csv)
        df_users = pd.read_csv(user_csv)
    except Exception as e:
        return f"❌ Lỗi đọc file: {e}. Đảm bảo bạn đã upload file lên Colab."

    results = []
    print(
        f"🚀 Bắt đầu phân tích {len(df_users)} người dùng và {len(df_songs)} bài hát...")

    # 2. VÒNG LẶP XỬ LÝ THEO CẶP (USER - SONG)
    for _, user in df_users.iterrows():
        for _, song in df_songs.iterrows():

            # --- PHẦN CỦA CƯỜNG: PHÂN TÍCH THẬT ---

            # Trích xuất đặc trưng bài hát (Content)
            # Không dùng ID để tính toán, lấy trực tiếp nhãn định tính
            song_features = {
                "song_id": int(song["song_id"]),
                "song_name": song["song_name"],
                "bpm": int(song["bpm"]),
                # Lấy từ dữ liệu thật: Energetic, Sad, Chill...
                "mood_detected": song["mood"],
                "genre": song["genre"],
                "is_high_tempo": True if song["bpm"] > 110 else False
            }

            # Trích xuất ngữ cảnh người dùng (Context)
            user_context = {
                "user_id": int(user["user_id"]),
                "user_name": user["user_name"],
                "location": user["location"],
                "activity": user["movement_speed"],
                "device": user["device"],
                # Suy luận logic: Nếu ở Gym hoặc đang Running -> Cần High Energy
                "inferred_vibe": "High Energy" if (user["location"] == "Gym" or user["movement_speed"] == "Running") else "Chill"
            }

            # 3. LOGIC SO KHỚP (MATCHING LOGIC) - Phải minh bạch
            # Bài hát khớp khi Mood bài hát tương ứng với nhu cầu người dùng
            match_conditions = [
                (song["mood"] == "Energetic" and user_context["inferred_vibe"]
                 == "High Energy"),
                (song["mood"] in ["Chill", "Sad", "Romantic"]
                 and user_context["inferred_vibe"] == "Chill")
            ]
            is_match = any(match_conditions)

            # 4. TỔNG HỢP GÓI DỮ LIỆU
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_report": f"Phân tích cho {user['user_name']} nghe bài {song['song_name']}",
                "content_analysis": song_features,
                "context_analysis": user_context,
                "is_recommended": is_match
            }
            results.append(entry)

    # --- XUẤT FILE JSON ---
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"✅ Đã xuất file thành công: {output_file}")
    return results
