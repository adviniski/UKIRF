import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import json

BASE_PATH = "results_baselines/"

files = []
files.append("ml-100k-gte.csv")

recallAtN = [1,5,10,20]
WINDOW_SIZE = 2000
K = 20

class WindowedRecall:
    def __init__(self, n, win_size):
        self.n = n
        self.win_size = win_size 
        self.hits = 0
        self.stream = 0 
        self.r_mean = 0.0
        self.recall_list = []
        self.ranks = []

    def update(self, rank):
        if rank < self.n:
            self.hits = 1
        else:
            self.hits = 0
            
        self.ranks.append(self.hits)
        
        if(len(self.ranks) > self.win_size):
            self.ranks.pop(0)
        recall = sum(self.ranks)/len(self.ranks)
        return recall

class MultipleRecall:
    def __init__(self):
        self.hits1 = 0
        self.hits5 = 0
        self.hits10 = 0
        self.hits20 = 0
        self.stream = 0
        
    def update(self, rank):
        if rank < 1:
            self.hits1 += 1
        if rank < 5:
            self.hits5 += 1
        if rank < 10:
            self.hits10 += 1
        if rank < 20:
            self.hits20 += 1
            
        self.stream += 1
        
    def score(self):
        recallat1 = round((self.hits1/self.stream),3)
        recallat5 = round((self.hits5/self.stream),3)
        recallat10 = round((self.hits10/self.stream),3)
        recallat20 = round((self.hits20/self.stream),3)
        
        return recallat1, recallat5, recallat10, recallat20

class Recall:
    def __init__(self, n):
        self.n = n
        self.hits = 0
        self.stream = 0
        
    def update(self, rank):
        if rank < self.n:
            self.hits += 1
        self.stream += 1
        
    def score(self):
        return float(self.hits/self.stream)


class Experiment():
    def __init__(self, filename, k):
        self.filename = filename
        self.evaluator = MultipleRecall()
        self.recall_value = 0.0

    def execute(self):
        with open(self.filename, 'r') as f:
            data = f.read().split(',')
            for rank in data:
                self.evaluator.update(int(rank))

        self.recall_value = self.evaluator.score()

        return self.recall_value

def evaluate_multiple_files():
    for K in recallAtN:
        results = {}
        with os.scandir(BASE_PATH) as it:
            for entry in it:
                if entry.is_dir():
                    print("Analyzing results for model %s"%(entry.name))
                    MODEL = entry.name
                    model_files = BASE_PATH+MODEL+'\\'
                    for file in files:
                        print("Analyzing results for dataset %s"%(file))
                        files_path = model_files+file+'\\'
                        list_files = [files_path+f for f in os.listdir(files_path) if re.search('dat|DAT', f)]
                        recall_values = []
                        for ind, filename in enumerate(list_files):
                            recall = Experiment(filename, K)
                            recall_values.append(recall.execute())
                            print("File %d analyzed"%(ind+1))
                            
                        if not file in results:
                            results[file] = {MODEL : recall_values}
                        else:
                            results[file][MODEL] = recall_values

        with open("results\\results_recallAt"+str(K)+".json", "w") as outfile:
            json.dump(results, outfile, indent = 4)

def baselines_analysis(list_files, MODEL):
    columns = ["Filename", "Model", "recall@1", "recall@5", "recall@10", "recall@20"]
    results = np.empty((len(list_files),len(columns)), dtype = object)

    for ind, file in enumerate(list_files):
        filename = file.replace(BASE_PATH+'/'+MODEL+'/',"")
        filename= filename.split(MODEL)[0]
        
        experiment = Experiment(file,K)
        recall_at1, recall_at5, recall_at10, recall_at20 = experiment.execute()

        results[ind,0] = filename
        results[ind,1] = MODEL
        results[ind,2] = recall_at1
        results[ind,3] = recall_at5
        results[ind,4] = recall_at10
        results[ind,5] = recall_at20

        print("Calculating recall at file %d"%(ind+1))
        data = pd.DataFrame(results, columns = columns)
    return data

def files_analysis(list_files, MODEL):
    columns = ["datasets", "RejectionMethod", "MaxRejection", "Model", "num_neg", "num_factors", "learning_rate", "reg_rate", "recallAt1", "recallAt5", "recallAt10", "recallAt20"]
    results = np.empty((len(list_files),len(columns)), dtype = object)

    for ind, file in enumerate(list_files):
        info = file.split(".dat")[0]
        info, rr = info.split("rr-")
        info, lr = info.split("lr-")
        info, num_factor = info.split("N_factor-")
        info, neg_items = info.split("-NegItems-")
        info, model = info.split("Model-")
        info, max_rejection = info.split("MaxRejection-")
        info, rejection_method = info.split("RejectionMethod-")
        index = info.index("-")
        execution, filename = info[0:index],info[index+1:len(info)]

        experiment = Experiment(file,K)
        recall_at1, recall_at5, recall_at10, recall_at20 = experiment.execute()

        dataset = filename.replace(BASE_PATH+'/'+MODEL+'/',"")

        results[ind,0] = str(dataset)
        results[ind,1] = str(rejection_method)
        results[ind,2] = str(max_rejection)
        results[ind,3] = str(model)
        results[ind,4] = int(neg_items)
        results[ind,5] = int(num_factor)
        results[ind,6] = float(lr)
        results[ind,7] = float(rr)
        results[ind,8] = float(recall_at1)
        results[ind,9] = float(recall_at5)
        results[ind,10] = float(recall_at10)
        results[ind,11] = float(recall_at20)

        print("Calculating recall at file %d"%(ind+1))

    data = pd.DataFrame(results, columns = columns)
    return data

def save_data(data, filename, results_name):

    datasets = data.datasets.unique()
    num_neg = sorted(data.num_neg.unique().astype(int))

    MaxRejection = data.MaxRejection.unique()
    RejectionMethod = data.RejectionMethod.unique()
    
    print(datasets)
    print(num_neg)
    print(MaxRejection)
    print(RejectionMethod)

    columns = ["dataset", "rejection_method", "max_rejection","negatives",  "recallAt1", "recallat5", "recallAt10", "recallAt20"]

    n_rows = len(datasets)*len(num_neg)*len(MaxRejection)*len(RejectionMethod)
    n_cols = len(columns)

    summarized = np.empty((n_rows,n_cols), dtype=object)

    index = 0
    for dataset in datasets:
        for neg in num_neg:
            for rejection in RejectionMethod:
                for max_r in MaxRejection:
                    ex = data.loc[(data.datasets == dataset) & (data.num_neg == neg) & (data.RejectionMethod == rejection) & (data.MaxRejection == max_r)]
                    
                    summarized[index,0] = dataset
                    summarized[index,1] = rejection
                    summarized[index,2] = max_r
                    summarized[index,3] = neg
                    summarized[index,4] = ex["recallAt1"].mean()
                    summarized[index,5] = ex["recallAt5"].mean()
                    summarized[index,6] = ex["recallAt10"].mean()
                    summarized[index,7] = ex["recallAt20"].mean()

                    index += 1

    print(summarized)
    
    data.to_excel(filename)

    results = pd.DataFrame(summarized, columns = columns)
    results.to_excel(results_name)

def evaluate_files(baseline = True):
    with os.scandir(BASE_PATH) as it:
        for entry in it:
            if entry.is_dir():
                data = None
                print("Analyzing results for model %s"%(entry.name))
                MODEL = entry.name
                files_path = BASE_PATH+'/'+MODEL+'/'
                list_files = [files_path+f for f in os.listdir(files_path) if re.search('dat|DAT', f)]
                if baseline:
                    data = baselines_analysis(list_files, MODEL)
                    filename = files_path+MODEL+"_results.xlsx"
                    data.to_excel(filename)
                else:
                    data = files_analysis(list_files, MODEL)
                    filename = files_path+MODEL+"_results.xlsx"
                    results_name = files_path+MODEL+"_summarized.xlsx"
                    save_data(data, filename, results_name)


def evaluate_one_file():
    BASE_PATH = 'C:/Users/david/OneDrive - Grupo Marista/doutorado/recommendation/tests/results/'
    MODEL = "TF-IDFBaseline"
    file = "SMDI-400k_max500repeated.csvTF-IDFBaseline1589394235.9539533.dat"

    filepath = BASE_PATH+MODEL+"/"+file
    recall = Experiment(filepath, K)
    recall_value = recall.execute()
    print("Result: %f %f %f %f"%(recall_value))

def evaluate_from_xlsx(file):
    data = pd.read_excel(file, index_col = None)
    results_name = file.replace(".xlsx", "_mean_std.xlsx")
    datasets = data.datasets.unique()
    num_neg = sorted(data.num_neg.unique().astype(int))

    MaxRejection = data.MaxRejection.unique()
    RejectionMethod = data.RejectionMethod.unique()

    columns = ["dataset", "rejection_method", "max_rejection","negatives",  "avg_recallAt1","std_recallAt1", "avg_recallAt5", "std_recallAt5", "avg_recallAt10", "std_recallAt10", "avg_recallAt20", "std_recallAt20"]

    n_rows = len(datasets)*len(num_neg)*len(MaxRejection)*len(RejectionMethod)
    n_cols = len(columns)

    summarized = np.empty((n_rows,n_cols), dtype=object)

    index = 0
    for dataset in datasets:
        for neg in num_neg:
            for rejection in RejectionMethod:
                for max_r in MaxRejection:
                    ex = data.loc[(data.datasets == dataset) & (data.num_neg == neg) & (data.RejectionMethod == rejection) & (data.MaxRejection == max_r)]
                    
                    summarized[index,0] = dataset
                    summarized[index,1] = rejection
                    summarized[index,2] = max_r
                    summarized[index,3] = neg
                    summarized[index,4] = ex["recallAt1"].mean()
                    summarized[index,5] = ex["recallAt1"].std()
                    summarized[index,6] = ex["recallAt5"].mean()
                    summarized[index,7] = ex["recallAt5"].std()
                    summarized[index,8] = ex["recallAt10"].mean()
                    summarized[index,9] = ex["recallAt10"].std()
                    summarized[index,10] = ex["recallAt20"].mean()
                    summarized[index,11] = ex["recallAt20"].std()

                    index += 1

    results = pd.DataFrame(summarized, columns = columns)
    results.to_excel(results_name)

def get_values(data, avg_recall, std_recall):
    data.sort_values(by = [avg_recall], ascending = False, inplace = True)
    avg_recall = data[avg_recall].values[0]
    std_recall = data[std_recall].values[0]
    neg = data.negatives.values[0]
    rem = data.max_rejection.values[0]
    
    if rem == "SuperiorLimitTotal":
        max_r = "SLT"
    elif rem == "SuperiorLimitUnique":
        max_r = "SLU"
    elif rem == "Q3Total":
        max_r = "Q3T"
    elif rem == "Q3Unique":
        max_r = "Q3U"

    model_data = "%.3f $\\pm$ %.4f & %d & %s"%(avg_recall, std_recall, neg, max_r)
    return model_data

def to_latex(file, model):
    data = pd.read_excel(file, index_col = None)
    
    datasets = data.dataset.unique()

    for d in datasets:
        if(d == "ml-100k-gte.csv"):
            random = data.loc[(data.dataset == d) & (data.rejection_method == "Random")]
            cosine = data.loc[(data.dataset == d) & (data.rejection_method == "COSINE")]
            tf_idf = data.loc[(data.dataset == d) & (data.rejection_method == "TF-IDF")]
            arm = data.loc[(data.dataset == d) & (data.rejection_method == "ARM")]

            #recall@1
            avg_recall, std_recall, at = "avg_recallAt1", "std_recallAt1", 1
            random_info = get_values(random, avg_recall, std_recall)
            cosine_info = get_values(cosine, avg_recall, std_recall)
            tf_idf_info = get_values(tf_idf, avg_recall, std_recall)
            arm_info = get_values(arm, avg_recall, std_recall)
            print("{\\multirow{4}{*}{\\rotatebox{90}{%s}}}"%(model))
            print(" & %d & %s & %s & %s & %s\\\\"%(at, random_info, cosine_info, tf_idf_info, arm_info))

            #recall@5
            avg_recall, std_recall, at = "avg_recallAt5", "std_recallAt5", 5
            random_info = get_values(random, avg_recall, std_recall)
            cosine_info = get_values(cosine, avg_recall, std_recall)
            tf_idf_info = get_values(tf_idf, avg_recall, std_recall)
            arm_info = get_values(arm, avg_recall, std_recall)
            print("& %d & %s & %s & %s & %s\\\\"%(at, random_info, cosine_info, tf_idf_info, arm_info))

            #recall@10
            avg_recall, std_recall, at = "avg_recallAt10", "std_recallAt10", 10
            random_info = get_values(random, avg_recall, std_recall)
            cosine_info = get_values(cosine, avg_recall, std_recall)
            tf_idf_info = get_values(tf_idf, avg_recall, std_recall)
            arm_info = get_values(arm, avg_recall, std_recall)
            print("& %d & %s & %s & %s & %s\\\\"%(at, random_info, cosine_info, tf_idf_info, arm_info))

            #recall@20
            avg_recall, std_recall, at = "avg_recallAt20", "std_recallAt20", 20
            random_info = get_values(random, avg_recall, std_recall)
            cosine_info = get_values(cosine, avg_recall, std_recall)
            tf_idf_info = get_values(tf_idf, avg_recall, std_recall)
            arm_info = get_values(arm, avg_recall, std_recall)
            print("& %d & %s & %s & %s & %s\\\\"%(at, random_info, cosine_info, tf_idf_info, arm_info))

if __name__ == "__main__":
    evaluate_files()
    summarized = False
    latex = False

    if summarized:
        evaluate_from_xlsx("results_negatives/BPRMF/BPRMF_results.xlsx")
        evaluate_from_xlsx("results_negatives/GMF/GMF_results.xlsx")
        evaluate_from_xlsx("results_negatives/NeuMF/NeuMF_results.xlsx")
        evaluate_from_xlsx("results_negatives/MLP/MLP_results.xlsx")

    if latex:
        to_latex("results_negatives/BPRMF/BPRMF_results_mean_std.xlsx", "BPRMF")
        to_latex("results_negatives/GMF/GMF_results_mean_std.xlsx", "GMF")
        to_latex("results_negatives/NeuMF/NeuMF_results_mean_std.xlsx", "MLP")
        to_latex("results_negatives/MLP/MLP_results_mean_std.xlsx", "NeuMF")