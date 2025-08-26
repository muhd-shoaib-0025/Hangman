# Shared Keras layers without Lambda ops. Safe to serialize/deserialize.
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention

@tf.keras.utils.register_keras_serializable(package="custom")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim=512, num_heads=1, ff_dim=1024, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.proj = Dense(self.embed_dim)
        self.self_attn = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim // self.num_heads)
        self.drop_attn = Dropout(self.dropout)
        self.norm_attn = LayerNormalization(epsilon=1e-6)
        self.ffn_1 = Dense(self.ff_dim, activation=tf.keras.activations.gelu)
        self.ffn_2 = Dense(self.embed_dim, activation=tf.keras.activations.gelu)
        self.drop_ffn = Dropout(self.dropout)
        self.norm_ffn = LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        x_proj = x if x.shape[-1] == self.embed_dim else self.proj(x)
        attn_out = self.self_attn(x_proj, x_proj, training=training)
        x_res1 = self.norm_attn(x_proj + self.drop_attn(attn_out, training=training))
        ffn_out = self.ffn_2(self.ffn_1(x_res1))
        return self.norm_ffn(x_res1 + self.drop_ffn(ffn_out, training=training))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(embed_dim=self.embed_dim, num_heads=self.num_heads, ff_dim=self.ff_dim, dropout=self.dropout))
        return cfg

@tf.keras.utils.register_keras_serializable(package="custom")
class GLUBlock(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1, residual=True, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.dropout_rate = float(dropout_rate)
        self.residual = bool(residual)
        self.linear_signal = Dense(self.units, activation="linear")
        self.linear_gate = Dense(self.units, activation="sigmoid")
        self.drop = Dropout(self.dropout_rate)
        self.add = tf.keras.layers.Add()
        self.mul = tf.keras.layers.Multiply()

    def call(self, x, training=None):
        signal = self.linear_signal(x)
        gate = self.linear_gate(x)
        y = self.mul([signal, gate])
        y = self.drop(y, training=training)
        return self.add([x, y]) if self.residual else y

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(units=self.units, dropout_rate=self.dropout_rate, residual=self.residual))
        return cfg

@tf.keras.utils.register_keras_serializable(package="custom")
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = int(max_len)
        self.embed_dim = int(embed_dim)
        self.pos_embed = tf.keras.layers.Embedding(self.max_len, self.embed_dim)

    def call(self, x):
        L = tf.shape(x)[1]
        pos = tf.range(L)
        pe = self.pos_embed(pos)[tf.newaxis, ...]
        return x + pe

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(max_len=self.max_len, embed_dim=self.embed_dim))
        return cfg

@tf.keras.utils.register_keras_serializable(package="custom")
class AttentionPool1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = Dense(1, activation=None)

    def call(self, x):
        # x: (B, L, D)
        s = self.score(x)                   # (B, L, 1)
        a = tf.nn.softmax(s, axis=1)        # (B, L, 1)
        return tf.reduce_sum(a * x, axis=1) # (B, D)

    def get_config(self):
        return super().get_config()
