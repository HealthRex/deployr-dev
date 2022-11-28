import pandas as pd
import numpy as np
import math
import json
import os

def readDataFromDirectory(dirname, experiment_name):
    #Read in index_str and index_num.json
    with open('./index_str.json') as f:
        index_str = json.load(f)
    with open('./index_num.json') as f:
        index_num = json.load(f)
    
    cohorts = ['0']
    cohorts += [index_num[str(i)] for i in range(127)]

    extractors = [[]]
    extractors += [index_str[str(i)] for i in range(127)]
 
    #Start reading in data from the dictionaries
    max_auroc = [0]
    max_drop = [0]
    for i in range(127):
        #1. Read in file from specific directory
        subset_dirname = os.path.join(dirname, experiment_name + f"_subset{i}" )
        filename = os.path.join(subset_dirname, "auc_max.json")
        with open(filename) as f:
            subset_dict = json.load(f)
        #2. Populate the max_auroc and max_drop lists
        max_auroc.append(subset_dict["max_auc"])
        max_drop.append(subset_dict["max_drop"])
    df_dict = {'Index': [i for i in range(128)],
               'Cohort': cohorts,
               'Extractors': extractors,
               'Max AUROC': max_auroc,
               'Max Drop': max_drop}
    df = pd.DataFrame(data=df_dict)
    return df

def shapley(df, col, n=None):
    if n is None:
        n = int(np.log2(df.shape[0]))
    for i in range(1, n+1):
        df_i = df[df["Cohort"].str.contains(str(i), na=False)]

        cohorts_w_i = df_i.Cohort.values.tolist()
        cohorts_no_i = [cohorts_w_i[j].replace(str(i), "") if cohorts_w_i[j] != str(i) else "0" for j in range(len(cohorts_w_i))]
        diffs = []
        coeffs = []

        for j in range(len(cohorts_no_i)):
            diff = df[df["Cohort"] == cohorts_w_i[j]][col].values[0] - df[df["Cohort"] == cohorts_no_i[j]][col].values[0]
            diffs.append(diff)
            
            if cohorts_no_i[j] == "0":
                s = 0
            else:
                s = len(cohorts_no_i[j])
            coeff = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            coeffs.append(coeff)
        shapley_i = np.sum([coeffs[j] * diffs[j] for j in range(len(cohorts_w_i))])
        print(f"Shapley Value at {col} for Player {df[df.Cohort == str(i)].Extractors.values[0]}: {shapley_i}")