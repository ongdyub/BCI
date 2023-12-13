import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dense, Dropout
import numpy as np
from .models import CTCEncoderLayer

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

class Encoder(Model):
    def __init__(self,
                 subsampleFactor=1,
                 nClasses=41,
                 num_layers = 6,
                 d_model = 512,
                 dff = 1024,
                 num_heads = 2,
                 dropout_rate = 0.1,
                 ):
        super(Encoder, self).__init__()
        
        self.d_model = d_model

        self.subsampleFactor = subsampleFactor
        
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.dense3 = tf.keras.layers.Dense(d_model, activation='relu')
        self.final_dense = tf.keras.layers.Dense(nClasses)
        
        self.attention = CTCEncoderLayer(num_layers, d_model, num_heads, dff)
        
        self.pos_encoding = positional_encoding(500, d_model)
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, states=None, training=False, returnState=False):
        
        x = tf.image.extract_patches(x[:, None, :, :],
                                        sizes=[1, 1, 32, 1],
                                        strides=[1, 1, 4, 1],
                                        rates=[1, 1, 1, 1],
                                        padding='VALID')
        x = tf.squeeze(x, axis=1)

        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)

        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        x = self.attention(x, training=training)
        
        x = self.final_dense(x, training=training)
        
        return x

    def getSubsampledTimeSteps(self, timeSteps):
        timeSteps = tf.cast((timeSteps - 32) / 4 + 1, dtype=tf.int32)
        return timeSteps
