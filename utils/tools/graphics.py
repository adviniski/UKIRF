import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from datetime import datetime


def plot_histogram(data):

    fig = plt.figure(figsize=(6,4))
    plt.rcParams.update({'font.size': 11})
    plt.rcParams.update({'font.weight': 'normal'})

    sb.kdeplot(data, color = (0.3,0.5,0.4,0.6), shade=True, Label='Items') 
      
    # Setting the X and Y Label 
    plt.xlabel("Index") 
    plt.ylabel('Probability Density') 

    plt.show()

def count_users_test(df):
    timestamps = df['timestamp'].values
    a = datetime.fromtimestamp(timestamps[0])
    b = datetime.fromtimestamp(timestamps[-1])

    mid = a + (b - a)/2

    date = datetime.timestamp(mid)
    print(mid)
    
    train_data =  df[df['timestamp'] <= date]

    test_data = df[df['timestamp'] > date]
    print("Gerando dados de usuários")

    users = test_data.user_id.values
    size = len(users)
    count = np.zeros(size)

    cumulative_users = np.zeros(df.user_id.nunique())
    train_users = train_data.user_id.unique()
    train_size = len(train_users)

    for i, u in enumerate(train_users):
        cumulative_users[i] = u

    n_users = train_size
    index = train_size

    for i in range(size):
        if users[i] not in cumulative_users:
            n_users += 1
            cumulative_users[index] = users[i]
            index += 1

        count[i] = n_users
        if(i % 10000 == 0):
            print(i)


    return count


def plot_data(recall, users_count):

    fig, ax1 = plt.subplots()
    SMALL_SIZE = 9
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)

    columns = recall.columns.values

    color = 'tab:red'
    right = u"\u2192"
    left = u"\u2190"
    
    ax1.set_xlabel('Old '+left+' Sample '+right+' New', fontweight='bold')
    ax1.set_ylabel('Recall@10', fontweight='bold')
    
    models = ["ISGD","IBPRMF","MLP","NeuMF","SVD","BPRMF","GMF","PopTop"]
    models = ["PopTop","GMF", "BPRMF","SVD","NeuMF","MLP","IBPRMF","ISGD"]
    
    colors = ["lightgreen", "forestgreen", "olive", "darkgreen", "seagreen", "greenyellow", "limegreen", "palegreen"]
    colors = ["cornflowerblue", "orangered", "green", "yellow", "magenta", "grey", "navy", "maroon"]

    markers = ["o", "v", "s", "P", "X", "*", "h", "d"]
    for i, c in enumerate(models):
    	ax1.plot(recall[c], label=c, 
                            color = colors[i], 
                            lw = 0.8, 
                            zorder=(i+2), 
                            marker=markers[i],
                            markersize=2,
                            linestyle = 'None',
                            markevery=800)


    ax1.tick_params(axis='y',labelsize=SMALL_SIZE)
    ax1.tick_params(axis='x', labelsize=SMALL_SIZE)
    ax1.set_ylim([0, 1])
    ax1.set_zorder(10)
    ax1.set_facecolor("none")
    
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), shadow=True, ncol=4, fontsize = 11)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_zorder(1)
    ax2.set_ylabel('Number of Users', fontweight='bold')  # we already handled the x-label with ax1
    ax2.plot(users_count, color ="red", label = "Nº of Users", lw = 2, zorder=1)
    ax2.tick_params(axis='y', labelsize=SMALL_SIZE)
    ax2.tick_params(axis='x', labelsize=SMALL_SIZE)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.7, 1.00), fontsize = 11)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def load_data(path="../data/datasets/SMDI-400k_max500repeated.csv", header=['user_id', 'item_id', 'rating', 'timestamp'], index_col = None, sep=","):
    print("Reading data...")
    df = pd.read_csv(path, sep=sep, names=header, index_col = index_col, engine='python')
    return df
