import faiss
import numpy as np


class VectorRetriever:
    def __init__(self, embedding_dim=32):
        self.embedding_dim = embedding_dim
        # Dùng Inner Product (IP) thay vì L2 vì vector đã được chuẩn hóa (Cosine Similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)

    def build_index(self, normalized_song_embeddings):
        """Chỉ nhận mảng vector numpy đã chuẩn hóa từ bên ngoài"""
        print("Nạp Vector kho nhạc vào bộ nhớ FAISS...")
        self.index.add(normalized_song_embeddings)
        print(f"Đã nạp {self.index.ntotal} bài hát.")

    def get_candidates(self, normalized_user_vector, top_k=50):
        """Truy vấn láng giềng gần nhất"""
        if self.index.ntotal == 0:
            raise ValueError("Index rỗng! Hãy chạy build_index trước.")

        distances, indices = self.index.search(normalized_user_vector, top_k)
        # Trả về mảng ID ứng viên và khoảng cách
        return indices[0], distances[0]
