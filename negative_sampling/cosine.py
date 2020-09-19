from negative_sampling.base import Base
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics.pairwise import cosine_similarity

class Cosine(Base):
	
	def __init__(self, data, num_users, num_neg_items):
		Base.__init__(self, data, num_users, num_neg_items)

	def get_default_neg(self, similarity_data):
		popularity = similarity_data.sum(axis=0)
		popularity.sort_values(ascending = False, inplace=True)
		ranked_popularity = popularity.index.values
		
		return ranked_popularity[0:self.N]

	def sampling_method(self):
		matrix = self.interaction_matrix()
		similarity_matrix = cosine_similarity(matrix)
		dataframe = pd.DataFrame(similarity_matrix, index = self.items, columns = self.items)

		return dataframe

	def get_users_neg(self, positive, similarity_data):
		negatives = list(set(self.items) - set(positive))
		matrix = similarity_data.loc[positive]
		matrix.drop(positive, axis=1, inplace = True)
		similarities = matrix.sum(axis = 0)
		similarities.sort_values(ascending = False, inplace=True)
		ranked_popularity = similarities.index.values

		return ranked_popularity[0:self.N]

	def __str__(self):
		return "COSINE"