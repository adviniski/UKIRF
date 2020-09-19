import numpy as np
import pandas as pd
import random


class Base(object):
	def __init__(self, data, num_users, num_neg_items):
		self.num_users = num_users
		self.num_neg_items = num_neg_items
		self.train_data = data
		self.users = None
		self.items = None
		self.num_neg_sample = 0
		self.neg_items_sample = {}
		self.N = 0
		self.all_index = None
		self.initialize()

	def initialize(self):
		self.users = sorted(self.train_data.user_id.unique())
		self.items = sorted(self.train_data.item_id.unique())

	def total_limit(self):
		data = self.train_data.copy()
		self.setNSuperiorLimit(data)

	def unique_limit(self):
		data = self.train_data.drop_duplicates(["user_id", "item_id"])
		self.setNSuperiorLimit(data)

	def q3_total_limit(self):
		df = self.train_data.copy()
		self.setNQ3(df)

	def q3_unique_limit(self):
		df = self.train_data.drop_duplicates(["user_id", "item_id"])
		self.setNQ3(df)

	def setNQ3(self, df):
		users = df.user_id.value_counts()
		q3 = np.percentile(users, 75)
		
		self.set_n(q3)

	def setNSuperiorLimit(self, df):
		users = df.user_id.value_counts()

		q1 = np.percentile(users, 25)
		q3 = np.percentile(users, 75)

		superior_limit = int(q3 + 1.5*(q3-q1))

		self.set_n(superior_limit)

	def set_neg_sample(self, value):
		""" 
			set number of negative samples per positive instance will be selected
		"""
		self.num_neg_sample = value
	
	def set_n(self, value):
		self.N = value

	def getItems(self):
		""" 
			return items in training data
		"""
		return self.items

	def getUsers(self):
		""" 
			return users in training data
		"""
		return self.users		
	
	def interaction_matrix(self):
		matrix = np.zeros(shape = (len(self.items),len(self.users)), dtype = float)
		train_users = self.train_data.user_id.values
		train_items = self.train_data.item_id.values

		train_data = list(zip(train_users, train_items))

		for u_id, i_id in train_data:
		    i = self.items.index(i_id)
		    u = self.users.index(u_id)

		    matrix[i,u] += 1

		return matrix

	def get_neg_items(self, Rejection):
		similarity_data = self.sampling_method()
		all_items = set(self.items)
		rejection = Rejection()
		self.N = rejection.execute(self.train_data)
		print("Default Number of Rejected items: %d"%(self.N))
		ranked_popularity = self.get_default_neg(similarity_data)
		print("Get users unknown items set")
		default_neg = list(all_items - set(ranked_popularity))
		
		for u in range(self.num_users):
		    if(u not in self.users):
		        neg_list = default_neg
		    else:
		        user_data = self.train_data.loc[(self.train_data.user_id == u)]
		        user_items = user_data.item_id.unique()
		        user_neg_items = all_items - set(user_items)
		        remove_from_neg = self.get_users_neg(user_items, similarity_data)
		        neg_list = list(user_neg_items - set(remove_from_neg))
		    
		    self.neg_items_sample[u] = neg_list

		return self.neg_items_sample

	def sampling_method(self):
		"""
			Return the data used to reject items from unknown item set
		"""
		pass

	def get_default_neg(self, similarity_data):
		"""
			Select the default items that will be removed for unknown users in the training set
			i.e, users that appear only in the test set
		"""
		pass

	def get_users_neg(self, positive, similarity_data):
		"""
			Return the negative items related to the positive observations of a known user
		"""
		pass

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

	def get_train_samples(self):
		size = len(self.train_data)
		size = size + size*(self.num_neg_sample)

		user_input = np.zeros((size), dtype=int)
		item_input = np.zeros((size), dtype=int)
		labels = np.zeros((size), dtype=float)

		index = 0

		users = self.train_data.user_id.values
		items = self.train_data.item_id.values

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

	def get_train_samples_pairwise(self):
		num_users, num_items = self.train_data.shape
		size = len(self.train_data)

		size = size + size*(self.num_neg_sample)

		user_input = np.zeros((size), dtype=int)
		item_input = np.zeros((size), dtype=int)
		neg_item_input = np.zeros((size), dtype=float)

		index = 0

		users = self.train_data.user_id.values
		items = self.train_data.item_id.values

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
	
	def aman(self):
		train_users_values = self.train_data.user_id.values
		train_items_values = self.train_data.item_id.values

		for user in users:
			user_items = list(set(self.items) - set(self.train_data[self.train_data.user_id == user]))

			train_users_values.extend([user]*len(user_items))
			train_items_values.extend(user_items)

		return np.array(train_users_values), np.array(train_items_values)

	