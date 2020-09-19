from negative_sampling.base import Base
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics.pairwise import cosine_similarity

class TF_IDF(Base):

	def __init__(self, data, num_users, num_neg_items):
		
		Base.__init__(self, data, num_users, num_neg_items)

	def get_default_neg(self, similarity_data):
		popularity = similarity_data.sum(axis=0)
		popularity.sort_values(ascending = False, inplace=True)
		ranked_popularity = popularity.index.values
		
		return ranked_popularity[0:self.N]
	
	def sampling_method(self):
		matrix = self.interaction_matrix()
		n_items = len(self.items)
		n_users = len(self.users)
		tf_idf_matrix = np.zeros(shape = (n_items, n_users), dtype = float)
		print("Calc TF_IDF values")
		t1 = time()

		non_zero_u = np.zeros(shape = (n_users), dtype = int)
		non_zero_i = np.zeros(shape = (n_items), dtype = int)

		for i in range(n_items):
			non_zero_i[i] = np.count_nonzero(matrix[i,:])

		for u in range(n_users):
			non_zero_u[u] = np.count_nonzero(matrix[:,u])

		for i in range(n_items):
			pos_i = non_zero_i[i]
			for u in range(n_users):
				if( matrix[i,u] > 0 ):
					tf = (matrix[i,u] / pos_i)
					idf = np.log(n_items / non_zero_u[u])
					tf_idf_matrix[i,u] = tf * idf
				else:
					tf_idf_matrix[i,u] = 0

		t2 = time()
		print("TF IDF time: %d"%(int(t2-t1)))
		
		similarity_matrix_skt = cosine_similarity(tf_idf_matrix)
		t3 = time()
		
		print("Cosine Similarity Time: %d"%(t3-t2))
		dataframe = pd.DataFrame(similarity_matrix_skt, index = self.items, columns = self.items)
		return dataframe 

	def get_users_neg(self, positive, similarity_data):
		matrix = similarity_data.loc[positive]
		matrix.drop(positive, axis=1, inplace = True)
		similarities = matrix.sum(axis = 0)
		similarities.sort_values(ascending = False, inplace=True)
		ranked_popularity = similarities.index.values

		return ranked_popularity[0:self.N]

	def __str__(self):
		return "TF-IDF"