import numpy as np


class FeatureFactory:
    """Độc lập với Processor. Nhận Feature Store (Dict) làm nguồn dữ liệu."""

    def __init__(self, user_feature_store, song_feature_store):
        self.user_store = user_feature_store
        self.song_store = song_feature_store

    def prepare_ranking_inputs(self, user_idx, candidate_indices):
        num_candidates = len(candidate_indices)
        if num_candidates == 0:
            return None, None

        # 1. Lấy dữ liệu User
        u_cat = self.user_store[user_idx]['cat']
        u_num = self.user_store[user_idx]['num']

        # 2. Lấy dữ liệu Songs bằng List Comprehension (Nhanh và Pythonic hơn)
        s_cat_batch = np.array([self.song_store[s_idx]['cat']
                               for s_idx in candidate_indices])
        s_num_batch = np.array([self.song_store[s_idx]['num']
                               for s_idx in candidate_indices])

        # 3. Broadcasting (Logic tile của Phong rất chuẩn)
        u_cat_batch = np.tile(u_cat, (num_candidates, 1))
        u_num_batch = np.tile(u_num, (num_candidates, 1))
        u_id_batch = np.full((num_candidates, 1), user_idx)
        s_id_batch = np.array(candidate_indices).reshape(-1, 1)

        # 4. Gộp ma trận (Feature Concatenation)
        X_rank_cat = np.hstack(
            [u_id_batch, u_cat_batch, s_id_batch, s_cat_batch]).astype('int32')
        X_rank_num = np.hstack([u_num_batch, s_num_batch]).astype('float32')

        return X_rank_cat, X_rank_num
