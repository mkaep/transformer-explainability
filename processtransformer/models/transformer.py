

from typing import Any

import tensorflow as tf
from keras import layers
from tensorflow.keras.models import Model

from processtransformer.models.multi_head_attention import MultiHeadAttention


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, train_attention=True, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim_py = embed_dim
        self.num_heads_py = num_heads
        self.ff_dim_py = ff_dim
        self.rate_py = rate
        self.train_attention_py = train_attention

        self.last_attn_scores = tf.Variable(0.0, shape=tf.TensorShape(None), trainable=False, validate_shape=False)

        layer_keys = ['att',
                      'ffn',
                      'layernorm_a',
                      'layernorm_b',
                      'dropout_a',
                      'dropout_b',
                      ]
        if set(layer_keys).issubset(set(kwargs.keys())):
            self.att = MultiHeadAttention.from_config(kwargs['att'])
            self.ffn = tf.keras.Sequential.from_config(kwargs['ffn'])
            self.layernorm_a = layers.LayerNormalization.from_config(kwargs['layernorm_a'])
            self.layernorm_b = layers.LayerNormalization.from_config(kwargs['layernorm_b'])
            self.dropout_a = layers.Dropout.from_config(kwargs['dropout_a'])
            self.dropout_b = layers.Dropout.from_config(kwargs['dropout_b'])
        else:
            self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, trainable=train_attention)
            self.ffn = tf.keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
            )
            self.layernorm_a = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm_b = layers.LayerNormalization(epsilon=1e-6)
            self.dropout_a = layers.Dropout(rate)
            self.dropout_b = layers.Dropout(rate)

    def call(self, inputs, training, mask=None,
             return_attention_scores=False):
        attn_output, attn_scores = self.att(inputs, inputs, return_attention_scores=True, attention_mask=mask)

        attn_output = self.dropout_a(attn_output, training=training)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output, training=training)
        normed = self.layernorm_b(out_a + ffn_output)
        if return_attention_scores:
            return normed, attn_scores
        return normed

    def get_config(self):
        config = {
            'embed_dim': self.embed_dim_py,
            'num_heads': self.num_heads_py,
            'ff_dim': self.ff_dim_py,
            'rate': self.rate_py,
            'train_attention': self.train_attention_py,
            # Layers
            'att': self.att.get_config(),
            'ffn': self.ffn.get_config(),
            'layernorm_a': self.layernorm_a.get_config(),
            'layernorm_b': self.layernorm_b.get_config(),
            'dropout_a': self.dropout_a.get_config(),
            'dropout_b': self.dropout_b.get_config(),
        }

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen_py = maxlen
        self.vocab_size_py = vocab_size
        self.embed_dim_py = embed_dim
        self.max_case_length = tf.Variable(maxlen, trainable=False, validate_shape=False)

        layer_keys = ['token_emb',
                      'pos_emb',
                      ]
        if set(layer_keys).issubset(set(kwargs.keys())):
            token_emb_kw = dict(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
            token_emb_kw.update(kwargs['token_emb'])
            self.token_emb = layers.Embedding.from_config(token_emb_kw)

            pos_emb_kw = dict(input_dim=maxlen, output_dim=embed_dim)
            pos_emb_kw.update(kwargs['pos_emb'])
            self.pos_emb = layers.Embedding.from_config(pos_emb_kw)
        else:
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        mask = self.token_emb.compute_mask(x)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
        # # Only the last dimension is multiplied with itself (outer dot product).
        # # I.e. for each word we transform the mask vector (1D) to a mask matrix (2D).
        # # The einsum means: Keep a and b, remove c and outer product of d and e.
        mask = tf.cast(mask, tf.int32)
        mask = tf.einsum('abcd,abce->abed', mask, mask)
        x = self.token_emb(x)
        return x + positions, mask

    def get_config(self):
        config = {
            'maxlen': self.maxlen_py,
            'vocab_size': self.vocab_size_py,
            'embed_dim': self.embed_dim_py,
            'token_emb': self.token_emb.get_config(),
            'pos_emb': self.pos_emb.get_config(),
        }

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Transformer(Model):
    # See https://keras.io/api/models/model/ (1 - function api, 2 - subclassing)
    def __init__(self, max_case_length, vocab_size, output_dim, name,
                 embed_dim=36, num_heads=4, ff_dim=64, train_attention=True,
                 **kwargs):
        super().__init__(name=name)
        self.max_case_length_py = max_case_length
        self.vocab_size_py = vocab_size
        self.output_dim_py = output_dim
        self.name_py = name
        self.embed_dim_py = embed_dim
        self.num_heads_py = num_heads
        self.ff_dim_py = ff_dim
        self.train_attention_py = train_attention

        self.last_attn_scores = tf.Variable(0.0, shape=tf.TensorShape(None), trainable=False, validate_shape=False)
        self.max_case_length = tf.Variable(max_case_length, trainable=False, validate_shape=False)

        layer_keys = ['token_and_pos_emb',
                      'transformer_block',
                      'pooling',
                      'dropout1',
                      'dense1',
                      'dropout2',
                      'dense2',
                      ]

        if set(layer_keys).issubset(set(kwargs.keys())):
            self.token_and_pos_emb = TokenAndPositionEmbedding.from_config(kwargs['token_and_pos_emb'])
            self.transformer_block = TransformerBlock.from_config(kwargs['transformer_block'])
            self.pooling = layers.GlobalAveragePooling1D.from_config(kwargs['pooling'])
            self.dropout1 = layers.Dropout.from_config(kwargs['dropout1'])
            self.dense1 = layers.Dense.from_config(kwargs['dense1'])
            self.dropout2 = layers.Dropout.from_config(kwargs['dropout2'])
            self.dense2 = layers.Dense.from_config(kwargs['dense2'])
        else:
            self.token_and_pos_emb = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)
            self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, train_attention=train_attention)
            self.pooling = layers.GlobalAveragePooling1D()
            self.dropout1 = layers.Dropout(0.1)
            self.dense1 = layers.Dense(ff_dim, activation="relu")
            self.dropout2 = layers.Dropout(0.1)
            self.dense2 = layers.Dense(output_dim, activation="linear")

    def call(self, inputs,
             training: Any = None,
             mask: Any = None,
             return_attention_scores=False):
        x, mask_emb = self.token_and_pos_emb(inputs)
        if mask is None:
            mask = mask_emb

        tb = self.transformer_block(x, mask=mask, return_attention_scores=return_attention_scores)
        if return_attention_scores:
            x, attn_scores = tb
        else:
            x = tb

        x = self.pooling(x)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)

        if return_attention_scores:
            # noinspection PyUnboundLocalVariable
            return x, attn_scores
        return x

    def get_config(self):
        config = {
            # Variables
            "max_case_length": self.max_case_length_py,
            "vocab_size": self.vocab_size_py,
            "output_dim": self.output_dim_py,
            "name": self.name_py,
            "embed_dim": self.embed_dim_py,
            "num_heads": self.num_heads_py,
            "ff_dim": self.ff_dim_py,
            "train_attention": self.train_attention_py,
            # Layers
            'token_and_pos_emb': self.token_and_pos_emb.get_config(),
            'transformer_block': self.transformer_block.get_config(),
            'pooling': self.pooling.get_config(),
            'dropout1': self.dropout1.get_config(),
            'dense1': self.dense1.get_config(),
            'dropout2': self.dropout2.get_config(),
            'dense2': self.dense2.get_config(),
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_next_activity_model(max_case_length, vocab_size, output_dim,
                            embed_dim=36, num_heads=4, ff_dim=64,
                            **transformer_kwargs,
                            ):
    return Transformer(max_case_length=max_case_length, vocab_size=vocab_size, output_dim=output_dim,
                       name="next_activity_transformer", embed_dim=embed_dim,
                       num_heads=num_heads, ff_dim=ff_dim, **transformer_kwargs)
