import tensorflow as tf
import random
import gc
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
from utils.graphics import *

user = os.getenv('username')

BASE_PATH = "C:\\Users\\"+user+"\\OneDrive - Grupo Marista\\SMDI-700k\\"

def optimize_dataframe(df, remove_unique = False):
    
    if remove_unique:
        users = df.user_id.value_counts()
        remove_users = users[users < 2].index.astype(int)
        df.drop(df[df.user_id.isin(remove_users)].index, inplace = True)
        
        items = df.item_id.value_counts()
        remove_items = items[items < 2].index.astype(int)
        
        print("Removing Users and Items with 1 interactions")
        while(len(remove_users) > 0):
            users = df.user_id.value_counts()
            remove_users = users[users < 2].index.astype(int)
            df.drop(df[df.user_id.isin(remove_users)].index, inplace = True)
            
            items = df.item_id.value_counts()
            remove_items = items[items < 2].index.astype(int)
            df.drop(df[df.item_id.isin(remove_items)].index, inplace = True)
    
    
    users = sorted(df.user_id.unique())
    items = sorted(df.item_id.unique())

    
    print("Optimizing Users Index")
    for u in users:
        if(u != users.index(u)):
            df['user_id'] = np.where(df['user_id'] == u, users.index(u), df['user_id'])
    
    
    print("Optimizing Items Index")
    for i in items:
        if(i != items.index(i)):
            df['item_id'] = np.where(df['item_id'] == i, items.index(i), df['item_id'])
    
    df.reset_index(inplace = True, drop = True)
    return df


def plot_histogram(data):
    #binwidth = 1
    #plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth),  density=True)
    #plt.title(title)
    #plt.show()
    # Plotting the KDE Plot 
    #sb.kdeplot(data, color = "r", shade=True, Label='Users')

    fig = plt.figure(figsize=(6,4))
    plt.rcParams.update({'font.size': 11})
    plt.rcParams.update({'font.weight': 'normal'})

    sb.kdeplot(data, color = (0.3,0.5,0.4,0.6), shade=True, Label='Items') 
      
    # Setting the X and Y Label 
    plt.xlabel("Index") 
    plt.ylabel('Probability Density') 

    plt.show()
    
def save_csv_file(df, filename):
    df.to_csv(filename, header=False, index=False)

def load_data_neg(path="../data/datasets/SMDI-400k_max500repeated.csv", filename = "SMDI-400k_max500repeated.csv"):
    print("Reading data...")
    df = pd.read_csv(path, names=["user_id", "item_id", "rating", "timestamp"], engine='python')
    
    df = optimize_dataframe(df)

    filepath = path.replace(".csv", "_optimized.csv")
    save_csv_file(df, filepath)
    
    user_data = sorted(df.user_id.values)
    item_data = sorted(df.item_id.values)

    #title = "User histogram for data set "+filename
    plot_histogram(user_data)
    plot_histogram(item_data)
    
    


if __name__ == '__main__':

    files = ["SMDI-700k_original_optimized.csv","SMDI-400k_max500repeated_optimized.csv","SMDI-400k_max200unique_optimized.csv"] 
    results = ["windowed_recall_SMDI-700k_original.csv","windoweddblp_recall_SMDI-400k_max500repeated.csv","windowed_recall_SMDI-400k_max200unique.csv"] 

    
    for i,file in enumerate(files):
        filepath = BASE_PATH + "data\\datasets\\" + file
        file_results = BASE_PATH + "\\results\\"+results[i]
        #dataset_statistics(filepath)
        data = load_data(path=filepath)

        users = count_users_test(data)
        
        result = load_data(path = file_results, header = None, index_col = "index")
        
        plot_data(result, users)
        
