import pandas as pd
import numpy as np 
import json
import time

def dataset_info():
    dataset = "data/datasets/SMDI-700k_original.csv"
    dataset_max500 = "data/datasets/SMDI-400k_max500repeated.csv"
    dataset_max200 = "data/datasets/SMDI-400k_max200unique.csv"

    users_data = "data/SMDI-700k_users.csv"
    items_data = "data/SMDI-700k_items.csv"

    products_data = "data/product_informations.csv"

    original = pd.read_csv(dataset, names = ["user_id", "item_id", "rating", "timestamp"])
    max500 = pd.read_csv(dataset_max500, names = ["user_id", "item_id", "rating", "timestamp"])
    max200 = pd.read_csv(dataset_max200, names = ["user_id", "item_id", "rating", "timestamp"])

    user_info = pd.read_csv(users_data)
    item_info = pd.read_csv(items_data)

    product_info = pd.read_csv(products_data)

    item_info.item_id = item_info.item_id.astype(int)
    item_info.section_id = item_info.section_id.astype(int)
    item_info.brand_id = item_info.brand_id.astype(int)

    product_info.item_id = product_info.item_id.astype(int)
    product_info.section_id = product_info.section_id.astype(int)
    product_info.brand_id = product_info.brand_id.astype(int)


    users = sorted(original.user_id.unique())
    items = sorted(original.item_id.unique())


    print("Optimizing Items Index")

    for i in items: 
        if(i != items.index(i)):
            original['item_id'] = np.where(original['item_id'] == i, items.index(i), original['item_id'])
            max500['item_id'] = np.where(max500['item_id'] == i, items.index(i), max500['item_id'])
            max200['item_id'] = np.where(max200['item_id'] == i, items.index(i), max200['item_id'])

            if(i not in item_info.item_id.values):
                print(i)
                index = product_info.index[product_info['item_id'] == i]
                print(index)
                

                
                item = product_info.iloc[index,:]
                print(item)
                if(item.iat[0,5] is 0.0):
                    if(item[3] is not 0.0):
                        item.iat[0,5] = item.iat[0,3]
                        item.iat[0,4] = (iitem.iat[0,3] + item.iat[0,6])/2
                    else:
                        item.iat[0,3] = item.iat[0,6]
                        item.iat[0,4] = item.iat[0,6]
                        item.iat[0,5] = item.iat[0,6]

                dataframe = pd.DataFrame(item,columns = product_info.columns)
                item_info.append(dataframe)

            item_info['item_id'] = np.where(item_info['item_id'] == i, items.index(i), item_info['item_id'])

    print("Optimizing Users Index")

    for u in users:
        if(u != users.index(u)):
            original['user_id'] = np.where(original['user_id'] == u, users.index(u), original['user_id'])
            max500['user_id'] = np.where(max500['user_id'] == u, users.index(u), max500['user_id'])
            max200['user_id'] = np.where(max200['user_id'] == u, users.index(u), max200['user_id'])
            user_info['user_id'] = np.where(user_info['user_id'] == u, users.index(u), user_info['user_id'])


    original.to_csv("data/datasets/SMDI-700k_original_optimezed.csv", header=False, index=False)
    max500.to_csv("data/datasets/SMDI-400k_max500repeated_optimezed.csv", header=False, index=False)
    max200.to_csv("data/datasets/SMDI-400k_max200unique_optimezed.csv", header=False, index=False)

    user_info.to_csv("data/SMDI-700k_users_optimized.csv", index=False)
    item_info.to_csv("data/SMDI-700k_items_optimized.csv", index=False)

if __name__ == "__main__":
    