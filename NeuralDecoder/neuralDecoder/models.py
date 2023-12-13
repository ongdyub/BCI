import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dense, Dropout
import numpy as np

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

def add_positional_encoding(inputs, pos_encoding):
    return inputs + pos_encoding

class GRU(Model):
    def __init__(self,
                 units,
                 weightReg,
                 actReg,
                 subsampleFactor,
                 nClasses,
                 bidirectional=False,
                 dropout=0.0,
                 nLayers=2,
                 conv_kwargs=None,
                 stack_kwargs=None):
        super(GRU, self).__init__()

        weightReg = tf.keras.regularizers.L2(weightReg)
        #actReg = tf.keras.regularizers.L2(actReg)
        actReg = None
        recurrent_init = tf.keras.initializers.Orthogonal()
        kernel_init = tf.keras.initializers.glorot_uniform()
        self.subsampleFactor = subsampleFactor
        self.bidirectional = bidirectional
        self.stack_kwargs = stack_kwargs
        self.pos_encoding = positional_encoding(500, 256)

        if bidirectional:
            self.initStates = [
                tf.Variable(initial_value=kernel_init(shape=(1, units))),
                tf.Variable(initial_value=kernel_init(shape=(1, units))),
            ]
        else:
            self.initStates = tf.Variable(initial_value=kernel_init(shape=(1, units)))

        self.conv1 = None
        if conv_kwargs is not None:
            self.conv1 = tf.keras.layers.DepthwiseConv1D(
                                                **conv_kwargs,
                                               padding='same',
                                               activation='relu',
                                               kernel_regularizer=weightReg,
                                               use_bias=False)

        self.rnnLayers = []
        for _ in range(nLayers):
            rnn = tf.keras.layers.GRU(units,
                                      return_sequences=True,
                                      return_state=True,
                                      kernel_regularizer=weightReg,
                                      activity_regularizer=actReg,
                                      recurrent_initializer=recurrent_init,
                                      kernel_initializer=kernel_init,
                                      dropout=dropout)
            self.rnnLayers.append(rnn)
        if bidirectional:
            self.rnnLayers = [tf.keras.layers.Bidirectional(rnn) for rnn in self.rnnLayers]
# TODO
############################################################################################################################################        
        self.inAttention = CTCEncoderLayer(1, 256, 2, 41) # pos + input attention
        # self.outAttention = CTCEncoderLayer(1, units, 4, 4096) # only output attention
        # self.outAttention = CTCEncoderLayer(1, units,8, 1024) # pos + output attention
        
        # self.midAttention = CTCEncoderLayer(1, units, 2, 41)
        # self.midAttention = CTCEncoderLayer(1, units, 4, 2048) # large
        # self.midAttention = CTCEncoderLayer(1, units, 4, 8192) # 4layer-large
        # self.hiddenAttention = CTCEncoderLayer(1, units,2, 4096)
        
        #mid out small
        # self.midAttention = CTCEncoderLayer(1, units, 2, units)
        # self.outAttention = CTCEncoderLayer(1, units, 2, units)
        # self.outAttention = CTCEncoderLayer(2, units, 2, 4096)
        # self.outAttention = CTCEncoderLayer(1, units, 2, 41)
        # self.midAttention = CTCEncoderLayer(4, units, 2, 512)
        self.dense = tf.keras.layers.Dense(nClasses)

    def call(self, x, states=None, training=False, returnState=False):
        batchSize = tf.shape(x)[0]
# TODO
############################################################################################################        
        # seq_len = tf.shape(x)[1]
        # x += self.pos_encoding[:, :seq_len, :]
        x = self.inAttention(x, training=training)
        
        if self.stack_kwargs is not None:
            x = tf.image.extract_patches(x[:, None, :, :],
                                         sizes=[1, 1, self.stack_kwargs['kernel_size'], 1],
                                         strides=[1, 1, self.stack_kwargs['strides'], 1],
                                         rates=[1, 1, 1, 1],
                                         padding='VALID')
            x = tf.squeeze(x, axis=1)

        if self.conv1 is not None:
            x = self.conv1(x)

        if states is None:
            states = []
            if self.bidirectional:
                states.append([tf.tile(s, [batchSize, 1]) for s in self.initStates])
            else:
                states.append(tf.tile(self.initStates, [batchSize, 1]))
            states.extend([None] * (len(self.rnnLayers) - 1))
# TODO
############################################################################################################        
        # seq_len = tf.shape(x)[1]
        # x += self.pos_encoding[:, :seq_len, :]
        
        # x = self.inAttention(x, training=training)
############################################################################################################        
        new_states = []
        if self.bidirectional:
            for i, rnn in enumerate(self.rnnLayers):
                x, forward_s, backward_s = rnn(x, training=training, initial_state=states[i])
                if i == len(self.rnnLayers) - 2:
                    if self.subsampleFactor > 1:
                        x = x[:, ::self.subsampleFactor, :]
                new_states.append([forward_s, backward_s])
        else:
            for i, rnn in enumerate(self.rnnLayers):
                x, s = rnn(x, training=training, initial_state=states[i])
                if i == len(self.rnnLayers) - 2:
                    if self.subsampleFactor > 1:
                        x = x[:, ::self.subsampleFactor, :]
# TODO
                # s = self.hiddenAttention(s, training=training)
                # x = self.midAttention(x, training=training)
                new_states.append(s)
# TODO
#########################################################################################################                
        # x = self.outAttention(x, training=training)
        x = self.dense(x, training=training)
#########################################################################################################
        if returnState:
            return x, new_states
        else:
            return x

    def getSubsampledTimeSteps(self, timeSteps):
        timeSteps = tf.cast(timeSteps / self.subsampleFactor, dtype=tf.int32)
        if self.stack_kwargs is not None:
            timeSteps = tf.cast((timeSteps - self.stack_kwargs['kernel_size']) / self.stack_kwargs['strides'] + 1, dtype=tf.int32)
        return timeSteps


class MultiHeadSelfAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = Dense(d_model)
        self.key_dense = Dense(d_model)
        self.value_dense = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs, training=training)
        key = self.key_dense(inputs, training=training)
        value = self.value_dense(inputs, training=training)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention, training=training)

        return output, attention_weights

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights


class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.att = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(d_model),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        self.build((None, d_model))

    def call(self, inputs, training=False):
        # print("[[[[[[[[[[[[[[[[[[[ENCDOER CALL]]]]]]]]]]]]]]]]]]]")
        # print(inputs.shape)
        attn_output, _ = self.att(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        # print(attn_output.shape)
        out1 = self.layernorm1(inputs + attn_output)
        # print(out1.shape)
        ffn_output = self.ffn(out1, training=training)
        # print(ffn_output.shape)
        ffn_output = self.dropout2(ffn_output, training=training)
        # print(ffn_output.shape)
        output = self.layernorm2(out1 + ffn_output)
        # print(output.shape)
        return output
    

class CTCEncoderLayer(Layer):
    def __init__(self, num_layers, d_model, num_heads, ff_dim):
        super(CTCEncoderLayer, self).__init__()

        self.transformer_blocks = [TransformerEncoderLayer(d_model, num_heads, ff_dim) for _ in range(num_layers)]

    def call(self, inputs, training=False):
        # x = self.embedding(inputs)
        # print("[[[[[[[[[[[[CALL START]]]]]]]]]]]]")
        x = inputs
        # print(x.shape)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training)

        # print("[[[[[[[[[[[[CALL END]]]]]]]]]]]]")
        # print(x.shape)
        return x