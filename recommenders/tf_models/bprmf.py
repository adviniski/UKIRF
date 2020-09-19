"""Implementation of Bayesain Personalized Ranking Model.
Reference: Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of
the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009..
"""

import tensorflow as tf
from time import time
import numpy as np
import random
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.models import Model
import pandas as pd

from recommenders.tf_models.base import Base


class BPRMF(Base):
    def __init__(self, num_user, num_item, learning_rate=0.001, reg_rate=0.01, epoch=500, batch_size=1024,
                    num_neg_sample = 30, num_neg_items = 1000, num_factor = 40, random_seed = 42):
        
        Base.__init__(self, num_user, num_item, learning_rate, reg_rate, epoch, batch_size,
                        num_neg_items, num_neg_sample,  num_factor, random_seed, model_name = "BPRMF")
    
    def get_model(self):
        
        user_input = Input(shape=(1,), dtype=tf.int32, name = 'user_input')
        item_input = Input(shape=(1,), dtype=tf.int32, name = 'item_input')
        item_neg_input = Input(shape=(1,), dtype=tf.int32, name = 'item_neg_input')
        
        
        Embedding_User = Embedding(self.num_user, self.num_factor,
                                       embeddings_regularizer = tf.keras.regularizers.l2(self.reg_rate),
                                       embeddings_initializer = tf.random_normal_initializer(stddev=0.01),
                                       input_length=1,
                                       name = "user_embedding")
        
        Embedding_Item = Embedding(self.num_item, self.num_factor,
                                       embeddings_regularizer = tf.keras.regularizers.l2(self.reg_rate),
                                       embeddings_initializer = tf.random_normal_initializer(stddev=0.01),
                                       input_length=1,
                                       name = "item_embedding")
        
        user_latent_factor = Flatten()(Embedding_User(user_input))
        item_latent_factor = Flatten()(Embedding_Item(item_input))
        neg_item_latent_factor = Flatten()(Embedding_Item(item_neg_input))
        
        pred_y = tf.reduce_sum(tf.keras.layers.multiply([user_latent_factor, item_latent_factor]), 1)
        pred_y_neg = tf.reduce_sum(tf.keras.layers.multiply([user_latent_factor, neg_item_latent_factor]), 1)
        

        model = Model(inputs = [user_input, item_input, item_neg_input], outputs = pred_y)
        
        def loss(y, y_neg):
            loss = - tf.reduce_sum(tf.math.log(tf.sigmoid(y - y_neg)))
            return loss
        
        model.add_loss(loss(pred_y, pred_y_neg))
        return model
    

    
    def run(self, train_data, test_data, neg_items, filename):
        print("Generating Model")
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            self.model = self.get_model()
            print("Compiling Model")
            self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate), loss = None)
        
        self.execute(train_data, test_data, neg_items, filename)

    def __str__(self):
        return "BPRMF"