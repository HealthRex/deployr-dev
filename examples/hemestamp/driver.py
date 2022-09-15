"""
Lab predictions and censoring
"""

import argparse
from google.cloud import bigquery
import os
import pandas as pd
import sys
import json

from healthrex_ml.cohorts import *
from healthrex_ml.trainers import *
from healthrex_ml.featurizers import *

# [EDIT] Authenticate to gcp project 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = (
    '/home/conorcorbin/.config/gcloud/application_default_credentials.json'
)
os.environ['GCLOUD_PROJECT'] = 'som-nero-phi-jonc101'

client = bigquery.Client()
parser = argparse.ArgumentParser(description='Build cohorts, featurize, train')
parser.add_argument(
    '--run_name',
    default='20220915_hemestamp_run',
    help='Ex: date of execution'
)
parser.add_argument(
    '--working_project_id',
    default='mining-clinical-decisions',
    help='Project id for temp tables'
)
parser.add_argument(
    '--working_dataset_name',
    default='conor_db',
    help='Bigquery dataset name where we store cohorts and features'
)
parser.add_argument(
    '--project_id',
    default='som-nero-phi-jonc101',
    help='project id that holds dataset used to create cohort and features'
)
parser.add_argument(
    '--dataset',
    default='shc_core_2021',
    help='dataset used to create cohort and features'
)
parser.add_argument(
    '--cohort_table_name',
    default='20220811_hemestamp_cohort',
    help='ex: just table name, project and dataset handled above'
)
parser.add_argument(
    '--feature_table_name',
    default='feature_matrix',
    help='just table name'
)
parser.add_argument(
    '--outpath',
    default='model_info',
    help='Out directory to write files to'
)
parser.add_argument(
    '--featurizer',
    default='BagOfWordsFeaturizer',
    help='Featurizer to use'
)
parser.add_argument(
    '--trainer',
    default='LightGBMTrainer',
    help='Trainer to use'
)
parser.add_argument(
    "--tasks",
    default=['label_hemestamp'],  # set of tasks (name of label column)
    nargs="*",
)
parser.add_argument(
    '--tfidf',
    action='store_true',
    help='whether to transform bow with tfidf'
)
args = parser.parse_args()

args_dict =  vars(args)
os.makedirs(f"./runs/{args.run_name}_{args.outpath}", exist_ok=True)
with open(f"./runs/{args.run_name}_{args.outpath}/command_args.json", 'w') as f:
    json.dump(args_dict, f)

# Feature config
FEATURE_CONFIG = {
    'Categorical': {
        'Sex': [{'look_back': None}],
        # 'Race': [{'look_back': None}],
        # 'Diagnoses': [{'look_back': None}], # None implies infinite look back
        # 'Medications': [{'look_back': 28}],
        # 'Procedures': [{'look_back': 28}]

    },
    'Numerical': {
        'Age': [{'look_back': None, 'num_bins': 5}],
        'LabResults': [{'look_back': 180, 'num_bins': 5}],
        # 'Vitals': [{'look_back': 3, 'num_bins': 5}]
    }
}

# Possible cohorts, featurizers and trainers
featurizers = {
    'BagOfWordsFeaturizer': BagOfWordsFeaturizer,
}
trainers = {
    'LightGBMTrainer': LightGBMTrainer,
    'BaselineModelTrainer': BaselineModelTrainer
}

cohort_table = (f"{args.working_project_id}.{args.working_dataset_name}."
                 f"{args.cohort_table_name}")
feature_table = (f"{args.working_project_id}.{args.working_dataset_name}."
                 f"{args.run_name}_{args.feature_table_name}")

if args.featurizer == 'BagOfWordsFeaturizer':
    featurizer = BagOfWordsFeaturizer(
        cohort_table_id=cohort_table,
        feature_table_id=feature_table,
        outpath=f"./runs/{args.run_name}_{args.outpath}",
        project=args.project_id,
        dataset=args.dataset,
        tfidf=args.tfidf,
        feature_config=FEATURE_CONFIG
    )
    print("Creating features")
    featurizer()  # Call to featurizer

# Train model
if args.trainer is not None:
    print("Training")

    trainer = trainers[args.trainer](
        working_dir=f"./runs/{args.run_name}_{args.outpath}"
    )
    for task in args.tasks:
        trainer(task)
