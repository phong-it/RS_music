import tensorflow as tf
from tensorflow.keras import layers, Model


class DeepFMRanker(Model):
    def __init__(self, cat_dims, embedding_dim=16):
        super(DeepFMRanker, self).__init__()
        self.embeddings = [layers.Embedding(input_dim=dim + 1, output_dim=embedding_dim)
                           for dim in cat_dims]

        self.dnn = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu')
        ])

        # ĐỔI: Chuyển activation thành 'linear' để dự đoán điểm số (Regression) thay vì xác suất (Sigmoid)
        self.final_dense = layers.Dense(1, activation='linear')

    def call(self, inputs):
        cat_inputs, num_inputs = inputs

        # --- Phần Embedding ---
        embedded = [self.embeddings[i](cat_inputs[:, i])
                    for i in range(len(self.embeddings))]
        concat_embed = tf.stack(embedded, axis=1)

        # --- FM Component ---
        sum_features_square = tf.square(tf.reduce_sum(concat_embed, axis=1))
        square_sum_features = tf.reduce_sum(tf.square(concat_embed), axis=1)
        fm_out = 0.5 * \
            tf.reduce_sum(sum_features_square -
                          square_sum_features, axis=1, keepdims=True)

        # --- Deep Component ---
        flatten_embed = layers.Flatten()(concat_embed)
        deep_in = tf.concat(
            [flatten_embed, tf.cast(num_inputs, tf.float32)], axis=1)
        deep_out = self.dnn(deep_in)

        # --- Combine ---
        concat_all = tf.concat([fm_out, deep_out], axis=1)
        return self.final_dense(concat_all)


class RankerTrainer:
    def __init__(self, cat_dims):
        self.model = DeepFMRanker(cat_dims)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='mse',  # ĐỔI: Dùng Mean Squared Error cho điểm số engagement
            metrics=['mae']
        )

    def fit(self, X_cat, X_num, y, epochs=10, batch_size=32):
        print("Đang huấn luyện DeepFM Ranker (MSE Loss)...")
        return self.model.fit([X_cat, X_num], y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
