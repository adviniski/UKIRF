import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from datetime import datetime
import random
import matplotlib.pyplot as plt

def parameters_tunning(path="../data/ml100k/movielens_100k.dat", header=['user_id', 'item_id', 'rating', 'timestamp'], test_size=0.5, sep="\t"):
    print("Reading data...")
    df = pd.read_csv(path, sep=sep, names=header, engine='python')
    
    df = df.drop_duplicates(["user_id", "item_id"])

    df.user_id = df.user_id.astype(int)
    df.item_id = df.item_id.astype(int)

    df = optimize_dataframe(df, True)
    df.sort_values(by = "timestamp", inplace = True)
    df.reset_index(inplace = True, drop=True)

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    users = df.user_id.unique()
    items = df.item_id.unique()
    test = np.empty((len(users),4), dtype=object)

    for row, user in enumerate(users):
        index = df.user_id.where(df.user_id==user).last_valid_index()

        test[row,0] = df.at[index,"user_id"]
        test[row,1] = df.at[index,"item_id"]
        test[row,2] = df.at[index,"rating"]
        test[row,3] = df.at[index,"timestamp"]
        df.drop(index = index,inplace=True)


    train_data = df
    train_data.to_excel("train_data.xlsx")
    test_data = pd.DataFrame(test, columns = header)
    test_data.to_excel("test_data.xlsx")

    print(test_data)

    return train_data, test_data, n_users, n_items

def count_users_day(df):

    timestamps = df['timestamp'].values
    dates = np.empty(len(timestamps),dtype = object)
    for i,t in enumerate(timestamps):
        dates[i] = datetime.fromtimestamp(t)

    df['dates'] = dates
    unique_date = np.unique(dates)

    count = []
    for d in unique_date:
        data = df[df['dates'] <= d]
        n = len(data.user_id.unique())
        count.append(n)


    fig = plt.figure(figsize=(6,4))
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'font.weight': 'normal'})
    plt.plot(list(range(len(count))),count)

    plt.ylabel('Number of Users', fontsize=12, fontweight='bold')
    plt.xlabel("Days", fontsize=12, fontweight='bold')
    
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
    print("Generating users data")

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


    fig=plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'font.weight': 'normal'})
    plt.axvline(x=43000, color='r', linestyle='-')
    plt.plot(list(range(len(count))),count, color = (0.3,0.5,0.4,0.6))

    plt.ylabel('Number of Users', fontsize=12, fontweight='bold')
    plt.xlabel("Test instances", fontsize=12, fontweight='bold')
    
    plt.show()

def count_users_instancia(df):

    print("Generating users data")

    users = df.user_id
    size = len(users)
    count = np.zeros(size)

    cumulative_users = np.zeros(df.user_id.nunique())
    n_users = 0
    index = 0

    for i in range(size):
        if users[i] not in cumulative_users:
            n_users += 1
            cumulative_users[index] = users[i]
            index += 1

        count[i] = n_users
        if(i % 10000 == 0):
            print(i)


    fig = plt.figure(figsize=(6,4))
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'font.weight': 'normal'})
    plt.plot(list(range(len(count))),count)

    plt.ylabel('Number of Users', fontsize=12, fontweight='bold')
    plt.xlabel("Days", fontsize=12, fontweight='bold')
    
    plt.show()

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
            df['user_id'].replace({u: users.index(u)}, inplace=True)
    
    
    print("Optimizing Items Index")
    for i in items:
        if(i != items.index(i)):
            df['item_id'].replace({i: items.index(i)}, inplace=True)
    
    df.reset_index(inplace = True, drop = True)
    return df

def load_rating_as_matrix(data):
    
    # Get number of users and items
    num_users, num_items = 0, 0
    for line in data.itertuples():
        u, i = int(line[0]), int(line[1])
        num_users = max(num_users, u)
        num_items = max(num_items, i)
        
    
    # Construct matrix
    mat = sp.dok_matrix((num_users+1, num_items+1), dtype = np.float32)
    for line in data.itertuples():
        user, item, rating = int(line[0]), int(line[1]), float(line[2])
        if (rating > 0):
            mat[user, item] = 1.0
            
    return mat

def get_sparse_data(data, n_users, n_items):
    row = []
    col = []
    rating = []
    
    for line in data.itertuples():
        u = line[1]
        i = line[2]
        row.append(u)
        col.append(i)
        rating.append(1)
    
    matrix = csr_matrix((rating, (row, col)), shape=(n_users, n_items), dtype = np.float64)
    return matrix

def remove_duplicates(df):
    
    df = df.drop_duplicates(["user_id", "item_id"])

    return df

def random_stratify(df, test_size = 0.5):
    
    train_data, test_data = train_test_split(df, test_size=test_size, stratify=df[["user_id"]], random_state = 0)
    
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    return train_data, test_data

def random_split(df, test_size = 0.5):

    train_data, test_data = train_test_split(df, test_size=test_size)
    
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    return train_data, test_data    

def temporal_split_by_instances(df, percentage = 0.5):
    
    split = int(df.shape[0]*percentage)
    train_data =  df.iloc[0:split,:]
    test_data = df.iloc[split:-1,:]

    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    return train_data, test_data

def temporal_split_by_timestamp(df, split = 2):
    timestamps = df['timestamp'].values
    a = datetime.fromtimestamp(timestamps[0])
    b = datetime.fromtimestamp(timestamps[-1])

    mid = a + (b - a)/split

    date = datetime.timestamp(mid)
    print(mid)

    train_data =  df[df['timestamp'] <= date]
    test_data = df[df['timestamp'] > date]

    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    return train_data, test_data

def save_csv_file(df, filename):
    df.to_csv(filename, header=False, index=False)
    
def load_data_neg(path="../data/datasets/SMDI-400k_max500repeated.csv", header=['user_id', 'item_id', 'rating', 'timestamp'], test_size=0.5, sep="\t", remove_unique = False):
    print("Reading data...")
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    df.user_id = df.user_id.astype(int)
    df.item_id = df.item_id.astype(int)

    df = optimize_dataframe(df, remove_unique)
    df.sort_values(by = "timestamp", inplace = True)
    
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    #print("Create stratified train and test sets")
    
    train_data, test_data = temporal_split_by_timestamp(df, split = 2)
    #train_data, test_data = random_split(df)
    print("Train instances:  - ",len(train_data)," Test Instances - ",len(test_data))
    print("Train: User - ",len(train_data.user_id.unique())," Items - ",len(train_data.item_id.unique()))
    print("Test: User - ", len(test_data.user_id.unique())," Items - ",len(test_data.item_id.unique()))

    return train_data, test_data, n_users, n_items
