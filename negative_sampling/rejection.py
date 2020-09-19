import numpy as np
import pandas as pandas


class TotalLimit(object):

	def execute(self, train_data):
		data = train_data.copy()
		return self.setNSuperiorLimit(data)

	def setNSuperiorLimit(self, df):
		users = df.user_id.value_counts()
		q1 = np.percentile(users, 25)
		q3 = np.percentile(users, 75)
		superior_limit = int(q3 + 1.5*(q3-q1))
		return superior_limit

	def __str__(self):
		return "SuperiorLimitTotal"


class UniqueLimit(object):

	def execute(self, train_data):
		data = train_data.drop_duplicates(["user_id", "item_id"])
		return self.setNSuperiorLimit(data)
	
	def setNSuperiorLimit(self, df):
		users = df.user_id.value_counts()
		q1 = np.percentile(users, 25)
		q3 = np.percentile(users, 75)
		superior_limit = int(q3 + 1.5*(q3-q1))
		return superior_limit

	def __str__(self):
		return "SuperiorLimitUnique"


class Q3Total(object):

	def execute(self, train_data):
		data = train_data.copy()
		return self.setNQ3(data)

	def setNQ3(self, df):
		users = df.user_id.value_counts()
		q3 = np.percentile(users, 75)

		return int(q3)

	def __str__(self):
		return "Q3Total"


class Q3Unique(object):

	def execute(self, train_data):
		data = train_data.drop_duplicates(["user_id", "item_id"])
		return self.setNQ3(data)

	def setNQ3(self, df):
		users = df.user_id.value_counts()
		q3 = np.percentile(users, 75)

		return int(q3)

	def __str__(self):
		return "Q3Unique"