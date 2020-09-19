from negative_sampling.base import Base
import numpy as np
import pandas as pd
from time import time
from utils.tools.association_rules import AssociationRules

class ARM(Base):
	"""docstring for TFIDF"""
	def __init__(self, data, num_users, num_neg_items, min_supp = 0.04, min_conf = 0.3):
		Base.__init__(self, data, num_users, num_neg_items)

		self.min_supp = min_supp
		self.min_conf = min_conf

	def verify_items(self,antecedents, consequents):
	    size = len(antecedents)
	    items_list = []
	    
	    for r in range(size):
	        items_a = list(antecedents[r])
	        items_c = list(consequents[r])
	        for a in items_a:
	            if(a not in items_list):
	                items_list.append(a)
	        for c in items_c:
	            if(c not in items_list):
	                items_list.append(c)

	    return items_list

	def sampling_method(self):
		df = self.train_data.drop_duplicates(["user_id", "item_id"])

		matrix = df.pivot('user_id', 'item_id','rating')
		matrix = matrix.fillna(0)
		matrix = matrix.astype("int32")
		AR = AssociationRules(matrix, self.min_supp, self.min_conf)
		rules = AR.execute()
		return rules

	def get_default_neg(self, sampling):
		antecedents = sampling["antecedents"].tolist()
		consequents = sampling["consequents"].tolist()
		associated_items = self.verify_items(antecedents, consequents)

		if(self.N > len(associated_items)):
			self.N = len(associated_items)
		    
		return associated_items[0:self.N]

	def get_users_neg(self, positive, similarity_data):
		associated_items_user = []
		
		antecedents = similarity_data["antecedents"].tolist()
		consequents = similarity_data["consequents"].tolist()
		
		size = len(antecedents)
		
		for item in positive:
		    for r in range(size):
		        items_a = list(antecedents[r])
		        items_c = list(consequents[r])

		        if item in items_a:
		            if len(items_a) > 1:
		                for i in items_a:
		                    if(i not in associated_items_user and i not in positive):
		                        associated_items_user.append(i)
		            for i in items_c:
		                if(i not in associated_items_user and i not in positive):
		                    associated_items_user.append(i)

		        elif item in items_c:
		            if len(items_c) > 1:
		                for i in items_c:
		                    if(i not in associated_items_user and i not in positive):
		                        associated_items_user.append(i)
		            for i in items_a:
		                if(i not in associated_items_user and i not in positive):
		                    associated_items_user.append(i)
		return associated_items_user
	
	def __str__(self):
   		return "ARM"

