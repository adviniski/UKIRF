import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras import Model
import os
from tensorflow.keras.optimizers import Adagrad, RMSprop, SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from time import time
import numpy as np
import random
import gc
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import csv
import copy

class Base(object):
    
    def __init__(self, num_user, num_item, learning_rate, reg_rate, epoch, batch_size, num_neg_items, num_neg_sample, num_factor, random_seed, model_name = "Model"):
        self.num_user = num_user
        self.num_item = num_item
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.num_factor = num_factor
        
        self.num_neg_items = num_neg_items
        self.num_neg_sample = num_neg_sample
        self.user_id = None  
        self.item_id = None  

        self.train_data = None
        self.test_data = None

        self.neg_items = None  
        self.test_users_vector = {}
        self.neg_items_sample = None  

        self.sampling_method = "random"
        self.max_rejection = "random"
        self.model = None
        self.PATH = None
        self.model_name = model_name
        self.result_file = "Model-"+str(self)+"-NegItems-"+str(self.num_neg_sample)+"N_factor-"+str(self.num_factor)+"lr-"+str(learning_rate)+"rr-"+str(reg_rate)+"time-"+str(time())+".dat"
        self.filename = None

        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    def setSamplingMethod(self, value):
        self.sampling_method = str(value).lower()
        return

    def setMaxRejectionApproach(self, value):
        self.max_rejection = str(value)
        return

    def get_train_instances_bprmf(self, train_data):
        num_users, num_items = train_data.shape
        size = len(train_data)
        
        size = size + size*(self.num_neg_sample)
        
        user_input = np.zeros((size), dtype=int)
        item_input = np.zeros((size), dtype=int)
        neg_item_input = np.zeros((size), dtype=float)
        
        index = 0

        users = train_data.user_id.values
        items = train_data.item_id.values

        train_samples = list(zip(users,items))

        neg_samples = self.copy_dict(self.neg_items_sample)

        for user_id, item_id in train_samples:

            if(self.num_neg_sample > 0):
                for t in range(self.num_neg_sample):
                    j = -1
                    len_user = len(neg_samples[user_id])
                    
                    if(len_user > 1):
                        j = random.choice(neg_samples[user_id])
                    elif(len_user is 1):
                        j = neg_samples[user_id][0]
                    else:
                        user_list = self.copy_list(self.neg_items_sample[user_id])
                        neg_samples[user_id] = user_list
                        j = random.choice(neg_samples[user_id])
                    
                    neg_samples[user_id].remove(j)

                    user_input[index] = user_id
                    item_input[index] = item_id
                    neg_item_input[index] = j
                    index += 1
            else:
                user_input[index] = user_id
                item_input[index] = item_id
                neg_item_input[index] = random.choice(self.neg_items_sample[user_id])
                index += 1
            
        return user_input, item_input, neg_item_input

    def copy_dict(self, data):
        copied = {}

        for key in data.keys():
            copied[key] = self.copy_list(data[key])

        return copied

    def copy_list(self, data_list):
        copied_list = []

        for i,value in enumerate(data_list):
            copied_list.append(value)

        return copied_list

    def get_train_samples(self, train_data):
        size = len(train_data)
        size = size + size*(self.num_neg_sample)
        
        user_input = np.zeros((size), dtype=int)
        item_input = np.zeros((size), dtype=int)
        labels = np.zeros((size), dtype=float)

        index = 0

        users = train_data.user_id.values
        items = train_data.item_id.values

        train_samples = list(zip(users,items))

        neg_samples = self.copy_dict(self.neg_items_sample)

        for user_id, item_id in train_samples:

            # positive instance
            user_input[index] = user_id
            item_input[index] = item_id
            labels[index] = 1

            index += 1
            if(self.num_neg_sample > 0):
                for t in range(self.num_neg_sample):
                    j = -1
                    len_user = len(neg_samples[user_id])
                    
                    if(len_user > 1):
                        j = random.choice(neg_samples[user_id])
                    elif(len_user is 1):
                        j = neg_samples[user_id][0]
                    else:
                        user_list = self.copy_list(self.neg_items_sample[user_id])
                        neg_samples[user_id] = user_list

                        if len(neg_samples[user_id]) is 0:
                            print("error")
                            print(self.neg_items_sample[user_id])

                        j = random.choice(neg_samples[user_id])
                    
                    neg_samples[user_id].remove(j)

                    user_input[index] = user_id
                    item_input[index] = j
                    labels[index] = 0
                    index += 1
            
        return user_input, item_input, labels

    def check_experiment(self):
        folder = "results_negatives/"+str(self)
        if not os.path.isdir(folder): # vemos de este diretorio ja existe
            return False
        else:
            file_path = folder+"/"+self.result_file
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    data = f.read().split(',')
                    if len(data) == len(self.test_data):
                        return True
        return False

    def execute(self, train_data, test_data, neg_items, filename):
        
        self.train_data = train_data
        self.test_data = test_data
        executed = self.check_experiment()
        if not executed:
            self.neg_items_sample = self.copy_dict(neg_items)
            self.filename = filename
            
            folder = "execution_time/"+str(self)
            if not os.path.isdir(folder): # vemos de este diretorio ja existe
                os.makedirs(folder) # aqui criamos a pasta caso nao exista
            
            with open(folder+"/"+str(self)+".txt", 'a') as f:
                f.write("Filename: "+str(self.filename)+"\n")
                f.write("Learning Rate: "+str(self.learning_rate)+"\n")
                f.write("Regularization Rate: "+str(self.reg_rate)+"\n")
                f.write("Start: "+str(time())+"\n")
                print("Start training")
                t1 = time()
                self.train()
                t2 = time()
                f.write("Training Time: "+str(t2-t1)+"\n")
                print("Start testing")
                self.test()
                f.write("Testing Time: "+str(time()-t2)+"\n")
                f.write("----------------------------------\n")
        else:
            print("Experiment already executed!!")

    def train(self):
        inputs = None
        output = None
        print("Get train samples")
        if(len(self.model.input_shape) is 2):
            user_input, item_input, label_input = self.get_train_samples(self.train_data)
            inputs = [user_input, item_input]
            output = label_input
            
        else:
            user_input, item_input, item_neg_input = self.get_train_instances_bprmf(self.train_data)
            inputs = [user_input, item_input, item_neg_input]
            output = None
            
        
        list_callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose = 2, restore_best_weights=True)
        ]
        
        print("Start Training")

        self.model.fit(inputs, #input
                          output, # labels
                          validation_split = 0.2,
                          batch_size = self.batch_size, 
                          epochs = self.epochs,
                          callbacks = list_callbacks,
                          verbose = 0, 
                          shuffle=True)
        
    def test(self):
        scores = {}
        users = self.test_data.user_id.unique()

        for u in users:
            user_vector = None
            if(len(self.neg_items_sample[u]) > self.num_neg_items):
                self.neg_items_sample[u] = random.sample(self.neg_items_sample[u], self.num_neg_items)
                user_vector =  np.full(self.num_neg_items, u, dtype = 'int32')
            else:
                user_vector =  np.full(len(self.neg_items_sample[u]), u, dtype = 'int32')

            pred_r = self.predict(user_vector, np.array(self.neg_items_sample[u]))
            scores[u] = list(pred_r)

        folder = "results_negatives/"+str(self)

        if not os.path.isdir(folder): # vemos de este diretorio ja existe
            os.makedirs(folder) # aqui criamos a pasta caso nao exista
        
        with open(folder+"/"+self.result_file, 'w+') as f:
            
            self.test_data.sort_values(by = "timestamp", inplace = True)
            
            users = self.test_data.user_id.values.astype(int)
            items = self.test_data.item_id.values.astype(int)

            pred_test = self.predict_on_batch(np.array(users), np.array(items))
            test_samples = list(zip(users, items, pred_test))

            index = 0

            for user, item, pred_rating in test_samples:
                rank = self.evaluation(user, item, pred_rating, scores[user])
                if index > 0:
                    f.write(",%s" % str(rank))
                else:
                    f.write("%s" % str(rank))

                index += 1

            f.close()
        
    def predict_on_batch(self, users, items):
        y_pred_probs = np.zeros([len(users)], dtype=float)

        BATCH_INDICES = np.arange(start=0, stop=len(users), step = self.batch_size)  # row indices of batches
        BATCH_INDICES = np.append(BATCH_INDICES, len(users))  # add final batch_end row
    
        if(len(self.model.input_shape) == 3):
            for index in np.arange(len(BATCH_INDICES) - 1):
                batch_start = BATCH_INDICES[index]  # first row of the batch
                batch_end = BATCH_INDICES[index + 1]  # last row of the batch
                y_pred_probs[batch_start:batch_end] = np.squeeze(self.model.predict_on_batch([users[batch_start:batch_end], items[batch_start:batch_end], items[batch_start:batch_end]]))
        else:
            for index in np.arange(len(BATCH_INDICES) - 1):
                batch_start = BATCH_INDICES[index]  # first row of the batch
                batch_end = BATCH_INDICES[index + 1]  # last row of the batch
                y_pred_probs[batch_start:batch_end] = np.squeeze(self.model.predict_on_batch([users[batch_start:batch_end], items[batch_start:batch_end]]))

        return y_pred_probs

    def predict_one_sample(self, user, item):
        inputs = None
        array_user = np.expand_dims(np.array([user]),0)
        array_item = np.expand_dims(np.array([item]),0)

        if len(self.model.input_shape) == 2:  
            inputs = [array_user, array_item]
        else:
            array_neg_item = np.expand_dims(np.array([0]),0)
            inputs = [array_user, array_item, array_neg_item]
        
        y_pred_probs = np.squeeze(self.model.predict_on_batch(inputs))
        return y_pred_probs

    def predict(self, users, items):
        size = len(items)
        y_pred_probs = np.zeros(size, dtype=float)
        inputs = None
        if len(self.model.input_shape) == 2:
            inputs = [users, items]
        else:
            neg_items = np.zeros(len(items), dtype = int)
            inputs = [users, items, neg_items]
        
        y_pred_probs = np.squeeze(self.model.predict_on_batch(inputs))
        return y_pred_probs

    def evaluation(self, user, item, rating, scores):
        users_items = list(self.neg_items_sample[user])
        predictions = list(scores)

        users_items.append(item)
        predictions.append(rating)

        neg_item_index = list(zip(users_items, predictions))

        ranked_list = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings = [r[0] for r in ranked_list]
        
        rank = pred_ratings.index(item)

        return rank

    def setResultFileName(self,filename, RejectionMethod, MaxRejection,repetition):
        self.filename = filename
        self.result_file = str(repetition)+"-"+filename+"RejectionMethod-"+str(RejectionMethod)+"MaxRejection-"+str(MaxRejection)+"Model-"+str(self)+"-NegItems-"+str(self.num_neg_sample)+"N_factor-"+str(self.num_factor)+"lr-"+str(self.learning_rate)+"rr-"+str(self.reg_rate)+".dat"