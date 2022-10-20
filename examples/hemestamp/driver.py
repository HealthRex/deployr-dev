"""
Driver to train a hemestamp model
ex: python driver.py --featurize --train
"""

import argparse
from google.cloud import bigquery
import json
import os
import pandas as pd
import sys
from healthrex_ml.cohorts import *
from healthrex_ml.trainers import *
from healthrex_ml.featurizers import *
from healthrex_ml.extractors import *
from healthrex_ml.evaluators import BinaryEvaluator

# Needs to point to your own application credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = (
    '/Users/conorcorbin/.config/gcloud/application_default_credentials.json'
)
os.environ['GCLOUD_PROJECT'] = 'som-nero-phi-jonc101'
client = bigquery.Client()

parser = argparse.ArgumentParser(description='Build cohorts, featurize, train')
parser.add_argument(
    '--experiment_name',
    default='20221020',
    help='Experiment name, prefix of all saved tables'
)
parser.add_argument(
    '--outpath',
    default='model_info',
    help='Out directory to write files to'
)
parser.add_argument(
    '--extractors',
    nargs='+',
    default=['AgeExtractor', 'RaceExtractor', 'EthnicityExtractor',
             'SexExtractor', 'PatientProblemExtractor',
             'LabOrderExtractor', 'ProcedureExtractor',
             'MedicationExtractor', 'LabResultBinsExtractor',
             'FlowsheetBinsExtractor'],
    help='List of extactors to use when'
)
parser.add_argument(
    '--featurizer',
    default='BagOfWordsFeaturizer',
    help='Featurizer to use'
)
parser.add_argument(
    '--trainer',
    default='LightGBMTrainer',
    help='Trainer to use (type of model)'
)
parser.add_argument(
    '--featurize',
    action='store_true',
    help='Whether to featurize'
)
parser.add_argument(
    '--train',
    action='store_true',
    help='Whether to train models'
)
args = parser.parse_args()

# Dump args to json
outpath = f"./{args.experiment_name}_hemestamp_{args.outpath}"
os.makedirs(outpath, exist_ok=True)
with open(os.path.join(outpath, 'args.json'), 'w') as f:
    json.dump(vars(args), f)

# Create dictionary of cohorts, featurizers and trainers
featurizers = {
    'BagOfWordsFeaturizer': BagOfWordsFeaturizer,
}
trainers = {
    'LightGBMTrainer': LightGBMTrainer,
}

cohort_table_id = f"mining-clinical-decisions.conor_db.20220811_hemestamp_cohort"
feature_table_id = f"mining-clinical-decisions.benchmarking.{args.experiment_name}_hemestamp_features"

# Create list of extactors
extractors = []
if 'AgeExtractor' in args.extractors:
    extractors.append(AgeExtractor(
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id
    ))
if 'RaceExtractor' in args.extractors:
    extractors.append(RaceExtractor(
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id
    ))
if 'SexExtractor' in args.extractors:
    extractors.append(SexExtractor(
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id
    ))
if 'EthnicityExtractor' in args.extractors:
    extractors.append(EthnicityExtractor(
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id
    ))
if 'PatientProblemExtractor' in args.extractors:
    extractors.append(PatientProblemExtractor(
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id
    ))
if 'LabOrderExtractor' in args.extractors:
    extractors.append(LabOrderExtractor(
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id
    ))
if 'ProcedureExtractor' in args.extractors:
    extractors.append(ProcedureExtractor(
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id
    ))
if 'MedicationExtractor' in args.extractors:
    extractors.append(MedicationExtractor(
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id
    ))
if 'LabResultBinsExtractor' in args.extractors:
    extractors.append(LabResultBinsExtractor(
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id,
        base_names=DEFAULT_LAB_COMPONENT_IDS
    ))
if 'FlowsheetBinsExtractor' in args.extractors:
    extractors.append(FlowsheetBinsExtractor(
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id,
        base_names=DEFAULT_FLOWSHEET_FEATURES
    ))

if args.featurize:
    print("Featurizing")
    featurizer = featurizers[args.featurizer](
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id,
        extractors=extractors,
        outpath=f"./{args.experiment_name}_hemestamp_{args.outpath}",
        tfidf=True
    )
    featurizer()

# Train model
if args.train:
    print("Training")
    trainer = trainers[args.trainer](
        working_dir=f"./{args.experiment_name}_hemestamp_{args.outpath}"
    )
    trainer(task='label_hemestamp')

# Evaluate model
yhats_path = os.path.join(f"./{args.experiment_name}_hemestamp_{args.outpath}",
                          'label_hemestamp_yhats.csv')
df_test = pd.read_csv(yhats_path)
evalr = BinaryEvaluator(
    outdir=f"./{args.experiment_name}_hemestamp_{args.outpath}")
evalr(df_test.labels, df_test.predictions)
