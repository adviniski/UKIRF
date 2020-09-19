import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from recommenders.baseline import PopTop, ARMBaseline, CosineBaseline, TF_IDFBaseline
from utils.load_data.load_data import *

if __name__ == '__main__':
    
    files = []

    files.append("SMDI-400k_max500repeated.csv")
    files.append("SMDI-400k_max200unique.csv")
    files.append("SMDI-700k_original.csv")
    files.append("ml-100k-gte.csv")

    for file in files:
        print("Start tests in the dataset %s"%(file))
        filepath = "../data/" + file
        train_data, test_data, n_user, n_item = load_data_neg(path=filepath, test_size=0.5, sep=",", remove_unique = False)
        
        pop_top = PopTop(num_users = n_user)
        pop_top.execute(train_data,test_data,file)

        cosine = CosineBaseline(num_users = n_user)
        cosine.execute(train_data,test_data,file)

        uf_iif = TF_IDFBaseline(num_users = n_user)
        uf_iif.execute(train_data,test_data,file)

        arm = ARMBaseline(num_users = n_user)
        arm.execute(train_data,test_data,file)