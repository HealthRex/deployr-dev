"""
Wrapper to execute various driver.py commands
"""
from itertools import chain, combinations
import json

#Extract all possible extractor subsets
extractors = [['AgeExtractor', 'RaceExtractor', 'EthnicityExtractor',
             'SexExtractor'], ['PatientProblemExtractor'], 
             ['LabOrderExtractor'], ['ProcedureExtractor'],
             ['MedicationExtractor'], ['LabResultBinsExtractor'],
             ['FlowsheetBinsExtractor']]

extractor_powerset = list(chain.from_iterable(combinations(extractors, r) for r in range(1, len(extractors) + 1)))

#Flatten lists
def convertElements(lst):
    new_lst = []
    for tup in lst:
        elem_lst = list(tup)
        flat_elem_lst = [item for sublist in elem_lst for item in sublist]
        new_lst.append(flat_elem_lst)
    return new_lst

extractor_subsets = convertElements(extractor_powerset)

# Set up mapping between index and extractor subsets and save this mapping
index = {i : extractor_subsets[i] for i in range(len(extractor_subsets))} 
index_file = './index.json'
with open(index_file, 'w') as f:
    json.dump(index, f)

# Set up commands
commands = []
for index, subset in index.items():
    subset_command = "python driver.py --build_cohort --cohort InpatientMortalityCohort " + \
    "--extract --featurize --train --evaluate --evaluator BinaryEvaluatorByTime "
    subset_command += f"--experiment_name 20221110_subset{index} "
    extractors_str = "--extractors " + ' '.join(subset)
    subset_command += extractors_str
    subset_command += " &"
    commands.append(subset_command)

for command in commands:
    print(command)
# Execute the commands [TODO]
# for command in commands:
#     os.system(command)
