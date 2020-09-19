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

class NeuMF(Base):
    def __init__(self, num_user, num_item, learning_rate=0.1, reg_rate=0.01, epoch=500, batch_size=1024,
                    num_neg_items = 1000, num_neg_sample = 2, num_factor = 40, random_seed = 42):
        
        Base.__init__(self, num_user, num_item, learning_rate, reg_rate, epoch, batch_size,
                        num_neg_items, num_neg_sample, num_factor, random_seed,  model_name = "NeuMF")

    def get_model(self, num_factor_mlp=64, hidden_dimension=8):
        
        user_input = Input(shape=(1,), dtype=tf.int32, name = 'user_input')
        item_input = Input(shape=(1,), dtype=tf.int32, name = 'item_input')
        
        
        Embedding_User = Embedding(self.num_user, self.num_factor,
                                       embeddings_regularizer = tf.keras.regularizers.l2(self.reg_rate),
                                       embeddings_initializer = tf.random_normal_initializer(mean = 0, stddev=0.01),
                                       input_length=1,
                                       name = "user_embedding")(user_input)
        
        Embedding_Item = Embedding(self.num_item, self.num_factor,
                                       embeddings_regularizer = tf.keras.regularizers.l2(self.reg_rate),
                                       embeddings_initializer = tf.random_normal_initializer(mean = 0, stddev=0.01),
                                       input_length=1,
                                       name = "item_embedding") (item_input)
        
        MLP_Embedding_User = Embedding(self.num_user, self.num_factor,
                                       embeddings_regularizer = tf.keras.regularizers.l2(self.reg_rate),
                                       embeddings_initializer = tf.random_normal_initializer(mean = 0, stddev=0.01),
                                       input_length=1,
                                       name = "mlp_user_embedding")(user_input)
        
        MLP_Embedding_Item = Embedding(self.num_item, self.num_factor,
                                       embeddings_regularizer = tf.keras.regularizers.l2(self.reg_rate),
                                       embeddings_initializer = tf.random_normal_initializer(mean = 0, stddev=0.01),
                                       input_length=1,
                                       name = "mlp_item_embedding")(item_input)
        
        
        user_latent_factor = Flatten()(Embedding_User)
        item_latent_factor = Flatten()(Embedding_Item)

        mlp_user_latent_factor = Flatten()(MLP_Embedding_User)
        mlp_item_latent_factor = Flatten()(MLP_Embedding_Item)
        
        _GMF = tf.keras.layers.multiply([user_latent_factor, item_latent_factor])
        
        vector = tf.keras.layers.concatenate([mlp_user_latent_factor, mlp_item_latent_factor])
        
        layer_1 = Dense(
            units = 32,
            kernel_initializer = tf.random_normal_initializer,
            activation = tf.nn.relu,
            kernel_regularizer = l2(self.reg_rate), 
            name = "layer_1")(vector)
        
        layer_2 = Dense(
            units = 16,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=l2(self.reg_rate), 
            name = "layer_2")(layer_1)

        layer_3 = Dense(
            units = 8,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=l2(self.reg_rate), 
            name = "layer_3")(layer_2)
        
        _MLP = Dense(
            units = 4,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=l2(self.reg_rate), 
            name = "layer_MLP")(layer_3)

        predict_vector =  tf.keras.layers.concatenate([_GMF, _MLP], axis=1)
        
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
        
        model = Model(inputs = [user_input, item_input], outputs = prediction)

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
        return "NeuMF"