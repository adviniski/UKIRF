from negative_sampling.base import Base

class Random(Base):
	"""docstring for TFIDF"""
	def __init__(self, data, num_users, num_neg_items):
		Base.__init__(self, data, num_users, num_neg_items)

	def get_default_neg(self, similarity_data):		
		return []

	def sampling_method(self):
		return None

	def get_users_neg(self, positive, similarity_data):
		return []

	def __str__(self):
		return "Random"