import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adagrad, RMSprop, SGD, Adam
from tensorflow.keras import initializers, backend as K
from tensorflow.keras.regularizers import l2
from time import time
import numpy as np
import random
import gc
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

from recommenders.tf_models.base import Base


class GMF(Base):
    def __init__(self, num_user, num_item, learning_rate=0.1, reg_rate=0.01, epoch=500, batch_size=1024,
                    num_neg_items = 1000, num_neg_sample = 2, num_factor = 40,  random_seed = 42):
        
        Base.__init__(self, num_user, num_item, learning_rate, reg_rate, epoch, batch_size,
                        num_neg_items, num_neg_sample, num_factor,  random_seed,  model_name = "GMF")
    
    def get_model(self, num_factor = 64):
        
        user_input = Input(shape=(1,), dtype=tf.int32, name = 'user_input')
        item_input = Input(shape=(1,), dtype=tf.int32, name = 'item_input')
        
        Embedding_User = Embedding(self.num_user, self.num_factor,
                                       embeddings_regularizer = tf.keras.regularizers.l2(self.reg_rate),
                                       embeddings_initializer = tf.random_normal_initializer(mean = 0, stddev=0.01),
                                       input_length=1, name = "user_embedding")
        
        Embedding_Item = Embedding(self.num_item, self.num_factor,
                                       embeddings_regularizer = tf.keras.regularizers.l2(self.reg_rate),
                                       embeddings_initializer = tf.random_normal_initializer(mean = 0, stddev=0.01),
                                       input_length=1, name = "item_embedding")
        
        
        user_latent_factor = Flatten()(Embedding_User(user_input))
        item_latent_factor = Flatten()(Embedding_Item(item_input))
        
        _GMF = tf.multiply(user_latent_factor, item_latent_factor)
        
        pred_y = tf.nn.sigmoid(tf.reduce_sum(_GMF, axis = 1))
        
        model = Model(inputs = [user_input, item_input], outputs = pred_y)
        return model
    
    
        
    def run(self, train_data, test_data, neg_items, filename):
        print("Generating Model")
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            self.model = self.get_model()
            print("Compiling Model")
            self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss = 'binary_crossentropy')
        
        self.execute(train_data, test_data, neg_items, filename)
    
    def __str__(self):
        return "GMF"