import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpmax
from mlxtend.frequent_patterns import fpgrowth
from decimal import Decimal
import gc


class AssociationRules(object):
	def __init__(self, dataset, support, confidence):
		self.data = dataset
		self.support = support
		self.confidence = confidence
		self.frequent_itemsets = None
		self.rules = None

	def encode_units(self,x):
	    if x <= 0:
	        return 0
	    if x >= 1:
	        return 1

	def get_frenquent_items(self):
		self.frequent_itemsets = apriori(self.data, min_support = self.support, use_colnames = True)

		return self.frequent_itemsets
		
	def get_rules(self):
		self.rules = association_rules(self.frequent_itemsets, metric = "confidence", min_threshold = self.confidence)
		self.rules.head()

	def execute_old(self):
		#self.data = self.data.applymap(self.encode_units)
		self.get_frenquent_items()
		self.get_rules()
		df = pd.DataFrame(self.rules)
		df.sort_values(by=["support","confidence","lift"], ascending=False, inplace=True)
		df.reset_index(drop=True, inplace=True)
		return df
	
	def execute(self):
		#self.data = self.data.applymap(self.encode_units)
		supp = self.support

		
		self.rules = pd.DataFrame(columns = ["antecedents", 
										"consequents", 
										"antecedent support",
										"consequent support",
										"support",
										"confidence",
										"lift",
										"leverage",
										"conviction"])
		num_rules = 0
		min_rules = 2
		while(num_rules < min_rules and supp > 0):
			try:
				if(len(self.data)*supp >= 5):
					frequent_itemsets = apriori(self.data, min_support = self.support, use_colnames = True)
					while len(frequent_itemsets) == 0:
						supp = self.suppUpdate(supp)
						
						if(len(self.data)*supp < 5):
							print("No rules founded!!")
							return self.rules
						else:	
							frequent_itemsets = apriori(self.data, min_support = self.support, use_colnames = True)
					
					rules = association_rules(self.frequent_itemsets, metric = "confidence", min_threshold = self.confidence)
					rules.head()

					rules["antecedents"] = rules["antecedents"].tolist()
					rules["consequents"] = rules["consequents"].tolist()

					df = pd.DataFrame(rules)
					df.sort_values(by=["support","confidence","lift"], ascending=False, inplace=True)
					df.reset_index(drop=True, inplace=True)
					num_rules = len(df)

					if(num_rules < min_rules):
						supp = self.suppUpdate(supp)
				else:
					print("No rules founded!!")
					return self.rules

			except MemoryError as error:
				print("MemoryError: "+str(error))
				gc.collect()
				return self.rules

			except ValueError as valueerror:
				print("ValueError: "+str(valueerror))
				gc.collect()
				return self.rules

			except Exception as exception:
				print("Exception Error:"+str(exception))
				gc.collect()
				return self.rules

		self.rules = rules
		return self.rules

	def suppUpdate(self, supp):
		int_places, dec_places = str(supp).split('.')
		num_dec = len(dec_places)
		dec = "0."
		for i in range(num_dec-1):
			dec = dec + "0"

		aux = dec + "1"
		supp = round((float(supp) - float(aux)),num_dec)
		if(supp <= 0.0):
			aux = dec + "09"
			supp = Decimal(aux).normalize()
			print("Aumenta casa decimal")
		print("Updating min support value: %s"%(str(supp)))

		return supp

	def to_list(self,x):
		list_x = list(x)
		name = ""
		for i,value in enumerate(list_x):
			name = self.products.loc[int(value),str(0)]
			list_x[i] = name

		return list_x