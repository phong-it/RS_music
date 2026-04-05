import numpy as np
import tensorflow as tf


class CoreRecommenderEngine:
    """Trái tim của hệ thống Cascade. Đầu ra là danh sách API thô."""

    def __init__(self, retriever, ranker_model, feature_factory, two_tower_model):
        self.retriever = retriever
        self.ranker = ranker_model
        self.factory = feature_factory
        self.two_tower = two_tower_model  # Cần để nhúng User trực tiếp

    def recommend(self, user_idx, user_raw_cat, user_raw_num, top_k_retrieval=50, top_n=5):
        # BƯỚC 1: TRUY XUẤT NHANH (Retrieval)
        # Biến User thành Vector tại chỗ để truy vấn FAISS
        u_vector = self.two_tower.user_tower(
            tf.convert_to_tensor(user_raw_cat),
            tf.convert_to_tensor(user_raw_num)
        )
        u_vector = tf.nn.l2_normalize(
            u_vector, axis=1).numpy().astype('float32')

        candidate_indices, _ = self.retriever.get_candidates(
            u_vector, top_k=top_k_retrieval)

        # BƯỚC 2: CHUẨN BỊ ĐẶC TRƯNG RANKING
        X_rank_cat, X_rank_num = self.factory.prepare_ranking_inputs(
            user_idx, candidate_indices)

        # BƯỚC 3: XẾP HẠNG TINH (Ranking)
        scores = self.ranker.predict(
            [X_rank_cat, X_rank_num], verbose=0).flatten()

        # BƯỚC 4: SẮP XẾP VÀ TRẢ VỀ PAYLOAD CHO API (Giao cho Người E)
        sorted_indices = np.argsort(scores)[::-1][:top_n]

        final_recommendations = []
        for idx in sorted_indices:
            final_recommendations.append({
                "song_id": int(candidate_indices[idx]),
                "predicted_score": float(scores[idx])
            })

        return final_recommendations  # Trả về data sạch
