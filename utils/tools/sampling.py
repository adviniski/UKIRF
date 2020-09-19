import numpy as np
import random


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

def get_train_samples(self, train_data, neg_items, num_neg_sample):
	size = len(train_data)
	size = size + size*(num_neg_sample)

	user_input = np.zeros((size), dtype=int)
	item_input = np.zeros((size), dtype=int)
	labels = np.zeros((size), dtype=float)

	index = 0

	users = train_data.user_id.values
	items = train_data.item_id.values

	train_samples = list(zip(users,items))
	neg_samples = self.copy_dict(neg_items)

	for user_id, item_id in train_samples:

	    # positive instance
	    user_input[index] = user_id
	    item_input[index] = item_id
	    labels[index] = 1

	    index += 1
	    if(num_neg_sample > 0):
	        for t in range(num_neg_sample):
	            j = -1
	            len_user = len(neg_samples[user_id])
	            
	            if(len_user > 1):
	                j = random.choice(neg_samples[user_id])
	            elif(len_user is 1):
	                j = neg_samples[user_id][0]
	            else:
	                user_list = self.copy_list(neg_items[user_id])
	                neg_samples[user_id] = user_list
					j = random.choice(neg_samples[user_id])
	            
	            neg_samples[user_id].remove(j)

	            user_input[index] = user_id
	            item_input[index] = j
	            labels[index] = 0
	            index += 1
	    
	return user_input, item_input, labels

def get_train_samples_pairwise(self, train_data, neg_items, num_neg_sample):
	num_users, num_items = train_data.shape
	size = len(train_data)

	size = size + size*(num_neg_sample)

	user_input = np.zeros((size), dtype=int)
	item_input = np.zeros((size), dtype=int)
	neg_item_input = np.zeros((size), dtype=float)

	index = 0

	users = train_data.user_id.values
	items = train_data.item_id.values

	train_samples = list(zip(users,items))

	neg_samples = self.copy_dict(neg_items)

	for user_id, item_id in train_samples:

	    if(num_neg_sample > 0):
	        for t in range(num_neg_sample):
	            j = -1
	            len_user = len(neg_samples[user_id])
	            
	            if(len_user > 1):
	                j = random.choice(neg_samples[user_id])
	            elif(len_user is 1):
	                j = neg_samples[user_id][0]
	            else:
	                user_list = self.copy_list(neg_items[user_id])
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
	        neg_item_input[index] = random.choice(neg_items[user_id])
	        index += 1
	    
	return user_input, item_input, neg_item_input