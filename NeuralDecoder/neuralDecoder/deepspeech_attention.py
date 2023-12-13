import tensorflow as tf
from tensorflow.keras import Model, constraints
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dense, Dropout
import numpy as np
from .models import CTCEncoderLayer

class DeepSpeechAttention(Model):
    def __init__(self,
                 nLayers=1,
                 weightReg=1e-5,
                 subsampleFactor=1,
                 nClasses=41,
                 dropout=0.15,
                 attentionLayer=1,
                 units=512,
                 n_head=8,
                 ff_dim=2048,
                 ):
        super(DeepSpeechAttention, self).__init__()

        weightReg = tf.keras.regularizers.L2(weightReg)
        #actReg = tf.keras.regularizers.L2(actReg)
        actReg = None
        kernel_init = tf.keras.initializers.glorot_uniform()
        
        self.subsampleFactor = subsampleFactor

        self.initStates_cell = tf.Variable(
                initial_value=kernel_init(shape=(1, units))
            )
        self.initStates_hidden = tf.Variable(
            initial_value=kernel_init(shape=(1, units))
        )

        self.lstmLayers = []
        for _ in range(nLayers):
            rnn = tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                return_state=True,
                kernel_regularizer=weightReg,
                activity_regularizer=actReg,
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                dropout=dropout,
            )
            self.lstmLayers.append(rnn)
        
        self.dense1 = tf.keras.layers.Dense(2048, kernel_constraint=constraints.MaxNorm(max_value=20.0), activation='gelu')
        self.dense2 = tf.keras.layers.Dense(2048, kernel_constraint=constraints.MaxNorm(max_value=20.0), activation='gelu')
        self.dense3 = tf.keras.layers.Dense(2048, kernel_constraint=constraints.MaxNorm(max_value=20.0), activation='gelu')
        
        # self.attention = CTCEncoderLayer(attentionLayer, units, n_head, ff_dim)
        
        self.final_dense = tf.keras.layers.Dense(nClasses)

    def call(self, x, states=None, training=False, returnState=False):
        batchSize = tf.shape(x)[0]
        
        x = tf.image.extract_patches(x[:, None, :, :],
                                        sizes=[1, 1, 32, 1],
                                        strides=[1, 1, 4, 1],
                                        rates=[1, 1, 1, 1],
                                        padding='VALID')
        x = tf.squeeze(x, axis=1)

        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)

        # LSTM
        if states is None:
            states = []
            state_cell = tf.tile(self.initStates_cell, [batchSize, 1])
            state_hidden = tf.tile(self.initStates_hidden, [batchSize, 1])
            states.append([state_cell, state_hidden])
            states.extend([None] * (len(self.lstmLayers) - 1))
            
        new_states = []
        for i, rnn in enumerate(self.lstmLayers):
            x, memory, carry = rnn(x, training=training, initial_state=states[i])
            new_states.append([memory, carry])
        
        # x = self.attention(x, training=training)
        
        x = self.final_dense(x, training=training)
        
        if returnState:
            return x, new_states
        else:
            return x

    def getSubsampledTimeSteps(self, timeSteps):
        timeSteps = tf.cast((timeSteps - 32) / 4 + 1, dtype=tf.int32)
        return timeSteps
