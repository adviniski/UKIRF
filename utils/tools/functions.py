import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def load_csv(path="../data/SMDI-400k_max500repeated.csv", header=['user_id', 'item_id', 'rating', 'timestamp'], index_col = None, sep=","):
	print("Reading data...")
	df = pd.read_csv(path, sep=sep, names=header, index_col = index_col, engine='python')
	return df

def dataset_statistics(data):

	print("Reading data...")
	df = data.copy()
	users = df.user_id.value_counts()

	q1 = np.percentile(users, 25)
	q2 = np.percentile(users, 50)
	q3 = np.percentile(users, 75)

	ls1 = float(q3 + 1.5*(q3-q1))
	print("Total before: %f"%(ls1))
	users_count = users[users <= 1000]

	q1 = np.percentile(users, 25)
	q2 = np.percentile(users, 50)
	q3 = np.percentile(users, 75)

	ls2 = float(q3 + 1.5*(q3-q1))
	print("Total after: %f"%(ls2))

	sns.set(style="whitegrid")
	fig = plt.figure(figsize=(6,4))
	plt.rcParams.update({'font.size': 12})
	plt.rcParams.update({'font.weight': 'normal'})
	ax = sns.boxplot(y=users_count)
	print(users_count)
	plt.show()

	df = data.drop_duplicates(["user_id", "item_id"])
	users = df.user_id.value_counts()

	q1 = np.percentile(users, 25)
	q2 = np.percentile(users, 50)
	q3 = np.percentile(users, 75)

	ls3 = float(q3 + 1.5*(q3-q1))
	print("Unique before: %f"%(ls3))

	users_count = users[users <= 1000]

	q1 = np.percentile(users, 25)
	q2 = np.percentile(users, 50)
	q3 = np.percentile(users, 75)

	ls4 = float(q3 + 1.5*(q3-q1))
	print("Unique after: %f"%(ls4))

	sns.set(style="whitegrid")

	fig = plt.figure(figsize=(6,4))
	plt.rcParams.update({'font.size': 12})
	plt.rcParams.update({'font.weight': 'normal'})

	print(users_count)

	ax = sns.boxplot(y=users_count)
	plt.show()

def dataset_info(data):
    fig = plt.figure(figsize=(6,4))
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'font.weight': 'normal'})
    
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    n_intances = df.shape[0]

    min_ratings = df.groupby("user_id").count()["item_id"].min()
    max_ratings = df.groupby("user_id").count()["item_id"].max()
    avg_ratings = int(df.groupby("user_id").count()["item_id"].mean())

    stat = pd.DataFrame({"count": df.groupby("user_id").count()["item_id"]}).reset_index(drop=True)
    stat.sort_values(by = "count", inplace = True)
    stat.reset_index(inplace = True, drop = True)

    plt.axhline(y=2.7, color='r', linestyle='-')
    plt.text(1, 2.9, r"$Log_{10}(500) = 2.69897$",  color='black', fontsize=10)
    plt.plot(np.log10(stat["count"]))

    plt.ylabel(r'$Log_{10}(N)$', fontsize=12, fontweight='bold')
    plt.xlabel("User's index", fontsize=12, fontweight='bold')
    
    plt.show()

    data = data.drop_duplicates(["user_id", "item_id"])
    
    n_instances_unique = data.shape[0]
    unique_min = data.groupby("user_id").count()["item_id"].min()
    unique_max = data.groupby("user_id").count()["item_id"].max()
    unique_avg = int(data.groupby("user_id").count()["item_id"].mean())

    unique = data.groupby("user_id").count()["item_id"].reset_index(drop=True)

    data = pd.DataFrame({"count": data.groupby("user_id").count()["item_id"]}).reset_index(drop=True)
    data.sort_values(by = "count", inplace = True)
    data.reset_index(inplace = True, drop = True)

    plt.text(1, 2.5, r"$Log_{10}(200) = 2.3010$",  color='black', fontsize=10)
    
    plt.axhline(y=2.3, color='r', linestyle='-')
    plt.plot(np.log10(data["count"]))

    plt.ylabel(r'$Log_{10}(N)$', fontsize=1, fontweight='bold')
    plt.xlabel("User's index", fontsize=12, fontweight='bold')
    #plt.title("Plot of moving average of Recall@"+str(K)+" values \nfor the test instances", fontsize=16, fontweight='bold', pad=10)
    #plt.xscale('log')
    plt.show()

    print("Number of Users: %d"%(n_users))
    print("Number of Items: %d"%(n_items))

    print("Number of ratings: %d"%(min_ratings))

    print("Min ratings: %d"%(min_ratings))
    print("Max ratings: %d"%(max_ratings))
    print("Avg ratings: %d"%(avg_ratings))

    print("Number of unique ratings: %d"%(n_instances_unique))

    print("Min unique ratings: %d"%(unique_min))
    print("Max unique ratings: %d"%(unique_max))
    print("Avg unique ratings: %d"%(unique_avg))

import json

def extract(paths):
    information_table = "Filename\t\tUsers\tItems\tRatings\tSparsity\n"
    filenames = []
    n_instances = []
    unique_ratings = []
    n_users = []
    l_users = []
    n_items = []
    l_items = []
    sparsities = []
    min_ratings = []
    max_ratings = []
    avg_ratings = []
    time_periods = []

    
    for path in paths:
        df = pd.read_csv(path, sep=',', names=['User', 'Item', 'Rating', 'Timestamp'], engine='python')
        unique = df.groupby(["User", "Item"]).size().reset_index()
        data = pd.DataFrame({"User": unique["User"], "Item": unique["Item"], "count": unique[0]}).reset_index()

        data = pd.DataFrame({'count' : data.groupby("User").count()["Item"]}).reset_index()

        data = data[data["count"] > 200]

        users = data["User"]

        df.drop(df[df.User.isin(users)].index, inplace = True)
        
        filename = path.split("\\")[-1]
        filenames.append(filename)
        
        n_instances.append(df.shape[0])
        
        users = df.groupby("User").count().index.values
        users.sort()
        n_user = users.shape[0]
        
        l_users.append(users[-1])
        n_users.append(n_user)
        
        items = df.groupby("Item").count().index.values
        items.sort()
        n_item = items.shape[0]
        
        l_items.append(items[-1])
        n_items.append(n_item)
        
        min_ratings.append(df.groupby("User").count()["Item"].min())
        max_ratings.append(df.groupby("User").count()["Item"].max())
        avg_ratings.append(int(df.groupby("User").count()["Item"].mean()))
        unique_rating = df.groupby(["User", "Item"]).count().shape[0]
        print("UR: {} EXP: {} User: {} Item: {} CR: {:.2f}%".format(unique_rating, n_user*n_item, n_user, n_item, (1 - unique_rating/(n_user*n_item))*100))
        sparsity = (1 - unique_rating/(n_user*n_item)) * 100
        sparsities.append(sparsity)
        unique_ratings.append(unique_rating/df.shape[0]*100)
    return pd.DataFrame({"Filename":filenames, "Size": n_instances, "Unique Rating": unique_ratings, "Users":n_users, "Last User ID": l_users, "Items":n_items, "Last Item ID": l_items, "Avg Rate":avg_ratings, "Min Rate":min_ratings, "Max Rate":max_ratings, "Sparsity":sparsities}).sort_values(by=["Size"])     