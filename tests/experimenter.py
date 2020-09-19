import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import random
import gc
from utils.load_data.load_data import *

from negative_sampling.arm import ARM
from negative_sampling.tf_idf import TF_IDF
from negative_sampling.cosine import Cosine
from negative_sampling.random import Random

from negative_sampling.rejection import TotalLimit, UniqueLimit, Q3Total, Q3Unique

from random import randint

class Experimenter(object):
    def __init__(self):
        self.computer = os.getenv('username')
        self.BASE_PATH = ""
        self.epochs = 100
        self.num_neg_items = 100
        self.negative_items = [1,2,3,4,5,6,7,8,9,10]
        self.files = None
        
        self.SAMPLING_APPROACHES = [ARM, TF_IDF, Cosine, Random]
        self.SAMPLING_METHOD = []
        
        self.MAX_REJECTION_METHODS = [TotalLimit, UniqueLimit, Q3Total, Q3Unique]
        self.MAX_REJECTION = []
        
        self.repetitions = 5
        self.MODEL = None
        self.count_gpus = 0

    def setModel(self, model):
        self.MODEL = model
        return

    def addSamplingApproach(self,value):
        if(value not in self.SAMPLING_APPROACHES):
            print("Sampling approach do not exist!!")

        elif (value in self.SAMPLING_METHOD):
            print("Sampling already informed!!")

        else:
            self.SAMPLING_METHOD.append(value)

    def addMaxRejection(self, value):
        if(value not in self.MAX_REJECTION_METHODS):
            print("Max rejection approach do not exist!!")

        elif (value in self.MAX_REJECTION):
            print("Max rejection approach already informed!!")

        else:
            self.MAX_REJECTION.append(value)

    def setParameterFiles(self, files):
        self.files = files
        return

    def config_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            self.count_gpus = len(gpus)
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    #tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')

                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def execute(self):		
        random_lists = list(range(1, 1024))
        for Sampling in self.SAMPLING_METHOD:
            
            for file in self.files.keys():
                print("Start tests in the dataset %s"%(file))
                filepath = self.BASE_PATH + "../data/" + file
                train_data, test_data, n_user, n_item = load_data_neg(path=filepath, test_size=0.5, sep=",", remove_unique = False)
                print("Start the Negative Rejection Approach")
                neg_sampling = Sampling(train_data.copy(), n_user, self.num_neg_items)
                for setN in self.MAX_REJECTION: 

                    neg_items = neg_sampling.get_neg_items(setN)
                    print("Max rejection approach %s"%(setN()))
                    
                    for r in range(self.repetitions):

                        for i, neg in enumerate(self.negative_items):

                            experimenter = None
                            random_seed = random.choice(random_lists)
                            paramenters = self.files[file]
                            experimenter = self.MODEL(n_user, n_item, learning_rate = paramenters["learning_rate"],
                                reg_rate = paramenters["reg_rate"], epoch = self.epochs, batch_size = paramenters["batch_size"],
                                num_neg_items = self.num_neg_items, num_neg_sample = neg, 
                                num_factor = paramenters["num_factor"], random_seed = random_seed)

                            experimenter.setSamplingMethod(neg_sampling)
                            experimenter.setResultFileName(str(file),neg_sampling,setN(), r)
                            print("Running - Method: %s; Number of Negatives: %d; Repetition: %d"%(experimenter, neg, r+1))
                            
                            experimenter.run(train_data.copy(), test_data.copy(), neg_items, file)
                            
                            del experimenter
                            
                            tf.keras.backend.clear_session()
                            gc.collect()

                    del neg_items