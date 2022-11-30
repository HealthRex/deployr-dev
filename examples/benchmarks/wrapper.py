"""
Wrapper to execute various driver.py commands
"""
from itertools import chain, combinations
import json
import os

#Extract all possible extractor subsets
extractors = [['AgeExtractor', 'RaceExtractor', 'EthnicityExtractor',
             'SexExtractor'], ['PatientProblemExtractor'], 
             ['LabOrderExtractor'], ['ProcedureExtractor'],
             ['MedicationExtractor'], ['LabResultBinsExtractor'],
             ['FlowsheetBinsExtractor']]

extractor_powerset = list(chain.from_iterable(combinations(extractors, r) for r in range(1, len(extractors) + 1)))
extractor_map_powerset = list(chain.from_iterable(combinations(range(1, len(extractors) + 1), r) for r in range(1, len(extractors) + 1)))

#Flatten lists
def convertElements(lst):
    new_lst = []
    for tup in lst:
        elem_lst = list(tup)
        flat_elem_lst = [item for sublist in elem_lst for item in sublist]
        new_lst.append(flat_elem_lst)
    return new_lst

extractor_subsets = convertElements(extractor_powerset)
extractor_maps = [list(ex_map) for ex_map in extractor_map_powerset]

# Set up mapping between index and extractor subsets and save this mapping
index_str = {i : extractor_subsets[i] for i in range(len(extractor_subsets))} 
index_num = {i : ''.join(str(n) for n in extractor_maps[i]) for i in range(len(extractor_map_powerset))}

with open("./index_str.json", "w") as f:
    json.dump(index_str, f)
with open("./index_num.json", "w") as f:
    json.dump(index_num, f)

# Set up commands
commands = []
date = "20221130"
cohort = "ThirtyDayReadmission"
cohort_abv = "readmit"
for index, subset in index_str.items():
    subset_command = f"python driver.py --cohort {cohort} " + \
    "--featurize --train --evaluate --evaluator BinaryEvaluatorByTime "
    subset_command += f"--outpath ./{date}_{cohort_abv}_subset{index} "
    subset_command += f"--experiment_name {date}_{cohort_abv} "
    extractors_str = "--extractors " + ' '.join(subset)
    subset_command += extractors_str
    subset_command += f" > ./log/{date}_{cohort_abv}_subset{index} 2>> ./log/{date}_{cohort_abv}_subset{index} &"
    commands.append(subset_command)

#Execute the commands
for i in range(len(commands)):
    os.system(commands[i])

