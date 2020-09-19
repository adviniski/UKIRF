import pandas as pd
import numpy as np
from time import time
import os


class PopTop(object):
    def __init__(self, num_users = None, num_neg_items = 100):
        self.num_neg_items = num_neg_items
        self.train_data = None
        self.test_data = None
        self.filename = None
        self.top_items = None
        self.model_name = "PopTop"

    def execute(self, train_data, test_data, filename):
        self.train_data = train_data
        self.test_data = test_data
        self.filename = filename

        self.train()
        self.test()

    def train(self):
        occurrences = self.train_data.item_id.value_counts()
        occurrences.sort_values(ascending=False)

        occurrences = occurrences.iloc[0:self.num_neg_items]

        self.top_items = np.empty(self.num_neg_items, dtype = int)
        i = 0
        for index, item in occurrences.iteritems():
            self.top_items[i] = index
            i += 1


    def test(self):
        t1 = time()
        folder = "results_baselines/"+self.model_name
        
        if not os.path.isdir(folder): # vemos de este diretorio ja existe
            os.makedirs(folder) # aqui criamos a pasta caso nao exista

        with open(folder+"/"+self.filename+self.model_name+str(time())+".dat", 'w+') as f:
            print("Começando a gerar resultados ...")
            self.test_data.sort_values(by = "timestamp", inplace = True)
            users = self.test_data.user_id.values
            items = self.test_data.item_id.values

            test_samples = list(zip(users, items))
            index = 0
            user, item = test_samples[index]
            rank = str(self.evaluate(user,item))
            f.write("%s" % str(rank))
            index += 1
            while (index < len(test_samples)):
                user, item = test_samples[index]
                rank = str(self.evaluate(user,item))
                f.write(",%s" % str(rank))
                index += 1
            f.close()
        t2 = time()
        print("Time to test: %d"%(t2-t1))

    def evaluate(self, user_id, item_id):
        if(item_id in self.top_items):
            rank = list(self.top_items).index(item_id)
        else:
            rank = self.num_neg_items

        return rank
