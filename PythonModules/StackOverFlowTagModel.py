#!/usr/bin/python
# -*- coding: utf-8 -*-
# importing packages

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import multiprocessing as mp
import heapq
from wordcloud import WordCloud
import gensim
import pickle
import random
import time
from sklearn.metrics import f1_score
import datetime
import os
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import MultiLabelBinarizer

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, \
    Dense, Dropout, Conv1D, MaxPooling1D, concatenate, TimeDistributed
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import History, ModelCheckpoint, \
    ReduceLROnPlateau, EarlyStopping, LearningRateScheduler

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

l2_alpha = 0


def multitask_loss(y_true, y_pred):

    # Avoid divide by 0

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # Multi-task loss

    return K.mean(K.sum(-y_true * K.log(y_pred) - (1 - y_true)
                  * K.log(1 - y_pred), axis=1))


def buildModel():
    """
    Creating Tag Classifier Model
    """

    tf.keras.backend.clear_session()

    Input_title = Input(shape=150, name='Input_Title')

    Emb_title = Embedding(input_dim=346929, output_dim=300,
                          name='Embedding_Layer')(Input_title)

    Lstm_title = LSTM(1024, name='LSTM_Layer', recurrent_dropout=0.2,
                      dropout=0.2,
                      kernel_regularizer=l2(l2_alpha))(Emb_title)

    flatten_title = Flatten(name='Flatten_lstm')(Lstm_title)

    currentOutput = Emb_title
    conv_filters = [
        2048,
        2048,
        1024,
        1024,
        512,
        512,
        ]
    for i in range(len(conv_filters)):
        conv = Conv1D(
            conv_filters[i],
            3,
            padding='same',
            activation='relu',
            kernel_initializer='glorot_uniform',
            name='Conv1D-' + str(i + 1),
            kernel_regularizer=l2(l2_alpha),
            )(currentOutput)
        maxPooling = MaxPooling1D(2, 2, name='MaxPooling1D-' + str(i
                                  + 1))(conv)
        currentOutput = maxPooling

    flatten_conv = Flatten(name='Flatten_Conv')(currentOutput)

    concatenateLayer = concatenate([flatten_title, flatten_conv],
                                   name='Concatenate')

    currentOutput = concatenateLayer
    nodes = [8192, 4096, 2048, 1024]
    for i in range(len(nodes)):
        dropout = Dropout(rate=0.4, name='DropOut_' + str(i
                          + 1))(currentOutput)
        dense = Dense(nodes[i], activation='relu',
                      kernel_initializer='glorot_uniform', name='Dense_'
                       + str(i + 1),
                      kernel_regularizer=l2(l2_alpha))(dropout)
        currentOutput = dense

    final_output = Dense(750, activation='sigmoid',
                         kernel_initializer='glorot_uniform',
                         name='OutputLayer',
                         kernel_regularizer=l2(l2_alpha))(currentOutput)

    inputs = Input_title

    model = Model(inputs=inputs, outputs=final_output)

    model.compile(optimizer='adam', loss=multitask_loss,
                  metrics=['accuracy'])

    # 'categorical_crossentropy'
    # multitask_loss

    return model