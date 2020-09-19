from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon, f_oneway, friedmanchisquare
import scikit_posthocs as sp
import os
import numpy as np
import pandas as pd
import math

from statistics import *

import json

# seed the random number generator
seed(1)

computer = os.getenv('username')

BASE_PATH = 'C:\\Users\\'+computer+'\\OneDrive - Grupo Marista\\SMDI-700k\\results\\'

models = ["SVD", "BPRMF", "GMF", "MLP", "NeuMF", "ISGD", "IBPRMF"]

data1 = None
data5 = None
data10 = None
data20 = None

with open(BASE_PATH+"results_recallAt1.json", 'r') as file1:
    data1 = json.load(file1)

with open(BASE_PATH+"results_recallAt1.json", "w") as outfile:
    json.dump(results, outfile, indent = 4)

with open(BASE_PATH+"results_recallAt5.json", 'r') as file5:
    data5 = json.load(file5)

with open(BASE_PATH+"results_recallAt10.json", 'r') as file10:
    data10 = json.load(file10)

with open(BASE_PATH+"results_recallAt20.json", 'r') as file20:
    data20 = json.load(file20)


def get_stat(results):
    avg = mean(results)
    std = stdev(results)
    return avg, std


def generate_latex_table():    

    print("\\begin{tabular}{|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Model}&\\textbf{RECALL@1}&\\textbf{RECALL@5}&\\textbf{RECALL@10}&\\textbf{RECALL@20}\\")
    print("\\hline")

   
    for dataset in data1.keys():
        print("\\hline")
        print("\\multicolumn{5}{|c|}{\\textbf{"+dataset+"}}\\\\")
        for model in models:
            avg1, std1 = get_stat(data1[dataset][model])
            avg5, std5 = get_stat(data5[dataset][model])
            avg10, std10 = get_stat(data10[dataset][model])
            avg20, std20 = get_stat(data20[dataset][model])
            print("\\hline")
            print("\\textbf{%s} & %.3f $\\pm$ %.4f & %.3f $\\pm$ %.4f & %.3f$\\pm$ %.4f & %.3f $\\pm$ %.4f \\\\"%(model, round(avg1,3),round(std1,4), round(avg5,3),round(std5,4), round(avg10,3),round(std10,4), round(avg20,3),round(std20,4)))
        print("\\hline")

#generate_latex_table()

def nemenyi():
    n = len(models)
    size = int(math.factorial(n)/(math.factorial(n-2)*math.factorial(2)))

    print(size)
    nemenyi_results = {}
    
    for dataset in data1.keys():
        print(dataset)

        results1 = data1[dataset]
        results5 = data5[dataset]
        results10 = data10[dataset]
        results20 = data20[dataset]

        nemenyi_results[dataset] = np.zeros(shape = (size, 5),dtype = object)
        index = 0
        matrix1 = np.zeros((30,7), dtype= float)
        matrix5 = np.zeros((30,7), dtype= float)
        matrix10 = np.zeros((30,7), dtype= float)
        matrix20 = np.zeros((30,7), dtype= float)

        res1 = None
        res5 = None
        res10 = None
        res20 = None
        
        for i, model in enumerate(models):
            matrix1[:,i] = results1[model]
            matrix5[:,i] = results5[model]
            matrix10[:,i] = results10[model]
            matrix20[:,i] = results20[model]

        res1 = sp.posthoc_nemenyi_friedman(matrix1)
        res5 = sp.posthoc_nemenyi_friedman(matrix5)
        res10 = sp.posthoc_nemenyi_friedman(matrix10)
        res20 = sp.posthoc_nemenyi_friedman(matrix20)

        col = 1
        for row in range(res1.shape[0]):
            for m in range(len(models) - col):
                p1,p5,p10,p20 = None,None,None,None
                p1 = round(res1.iloc[row, col+m], 4)
                p5 = round(res5.iloc[row, col+m], 4)
                p10 = round(res10.iloc[row, col+m], 4)
                p20 = round(res20.iloc[row, col+m] , 4)
                
                print("\\hline")
                if(p1 < 0.05):
                    p1 = "\\textit{%f}"%(p1)
                else:
                    p1 = "%f"%(p1)

                if(p5 < 0.05):
                    p5 = "\\textit{%f}"%(p5)
                else:
                    p5 = "%f"%(p5)

                if(p10 < 0.05):
                    p10 = "\\textit{%f}"%(p10)
                else:
                    p10 = "%f"%(p10)

                if(p20 < 0.05):
                    p20 = "\\textit{%f}"%(p20)
                else:
                    p20 = "%f"%(p20)

                comp = "%s vs %s"%(models[row], models[col+m])
                
                nemenyi_results[dataset][index,0] = comp
                nemenyi_results[dataset][index,1] = p1
                nemenyi_results[dataset][index,2] = p5
                nemenyi_results[dataset][index,3] = p10
                nemenyi_results[dataset][index,4] = p20
                index+=1
                print("\\textbf{%s vs %s} & %s & %s & %s & %s \\\\"%(models[row], models[col+m], p1, p5, p10, p20))
            col += 1

    return nemenyi_results

def nonparametric_test():


    n = len(models)
    size = int(math.factorial(n)/(math.factorial(n-2)*math.factorial(2)))

    wilcoxon_results = {}
    function = wilcoxon
    for dataset in data1.keys():
        comparisons = []

        results1 = data1[dataset]
        results5 = data5[dataset]
        results10 = data10[dataset]
        results20 = data20[dataset]

        wilcoxon_results[dataset] = np.zeros((size,5),dtype = object)
        index = 0

        #print("\\hline")
        #print("\\multicolumn{5}{|c|}{\\textbf{"+dataset+"}}\\\\")
        
        for model1 in models:
            for model2 in models:
                if model1 != model2 and (model1,model2) not in comparisons and (model2,model1) not in comparisons:
        
                    stat1, p1 = function(results1[model1], results1[model2])
                    stat5, p5 = function(results5[model1], results5[model2])
                    stat10, p10 = function(results10[model1], results10[model2])
                    stat20, p20 = function(results20[model1], results20[model2])

                    comparisons.append((model1,model2))
                    #print("\\hline")
                    if(p1 < 0.05):
                        p1 = "\\textit{%f}"%(p1)
                    else:
                        p1 = "%f"%(p1)

                    if(p5 < 0.05):
                        p5 = "\\textit{%f}"%(p5)
                    else:
                        p5 = "%f"%(p5)

                    if(p10 < 0.05):
                        p10 = "\\textit{%f}"%(p10)
                    else:
                        p10 = "%f"%(p10)

                    if(p20 < 0.05):
                        p20 = "\\textit{%f}"%(p20)
                    else:
                        p20 = "%f"%(p20)

                    comp = "%s vs %s"%(model1, model2)

                
                    wilcoxon_results[dataset][index,0] = comp
                    wilcoxon_results[dataset][index,1] = p1
                    wilcoxon_results[dataset][index,2] = p5
                    wilcoxon_results[dataset][index,3] = p10
                    wilcoxon_results[dataset][index,4] = p20
                    index+=1
                    
                    #print("\\textbf{%s vs %s} & %s & %s & %s & %s \\\\"%(model1, model2, p1, p5, p10, p20))
                    
                    
        #print("\\hline")
    return wilcoxon_results


generate_latex_table()
neme = nemenyi()
"""wilco = nonparametric_test()
for key in neme.keys():

    test1 = neme[key]
    test2 = wilco[key]

    print(key)
    print()
    
    
    for line in range(len(test1)):
        p1t1 = test1[line,1]
        p1t5 = test1[line,2]
        p1t10 = test1[line,3]
        p1t20 = test1[line,4]

        p2t1 = test2[line,1]
        p2t5 = test2[line,2]
        p2t10 = test2[line,3]
        p2t20 = test2[line,4]

        
        print("%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\"%(test1[line,0], p1t1, p2t1, p1t5, p2t5, p1t10, p2t10, p1t20, p2t20))

        print("\\hline")
    print()
    print()
    print()
"""
