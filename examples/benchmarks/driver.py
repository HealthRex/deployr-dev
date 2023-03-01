"""
Example driver.py to train and evaluate three benchmark tasks
1. Inpatient Mortality
2. 30 day readmission
3. Long length of stay
"""

import argparse
# from msilib.schema import Binary
from google.cloud import bigquery
import os
import pandas as pd
import sys

from healthrex_ml.cohorts import *
from healthrex_ml.trainers import *
from healthrex_ml.featurizers import *
from healthrex_ml.extractors import *
from healthrex_ml.evaluators import BinaryEvaluator, BinaryEvaluatorByTime

DATASET_NAME = "raphael_honors"

# Authenticate to gcp project [TODO -- change to point to your credentials file]
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = (
    #'C:\\Users\\Raphael\\AppData\\Roaming\\gcloud\\application_default_credentials.json' #On Local
    '/home/Raphael/.config/gcloud/application_default_credentials.json'                   #On cloud instance
)
os.environ['GCLOUD_PROJECT'] = 'som-nero-phi-jonc101'
client = bigquery.Client()

parser = argparse.ArgumentParser(description='Build cohorts, featurize, train')
parser.add_argument(
    '--experiment_name',
    default='20221115',
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
    '--cohort',
    default='InpatientMortalityCohort',
    help='Cohorts to use'
)
parser.add_argument(
    '--featurizer',
    default='BagOfWordsFeaturizer',
    help='Featurizer to use'
)
parser.add_argument(
    '--trainer',
    default='LightGBMTrainer',
    help='Featurizer to use'
)
parser.add_argument(
    '--build_cohort',
    action='store_true',
    help='Whether to build cohort'
)
parser.add_argument(
    '--extract',
    action='store_true',
    help='Whether to call extractors, if False assumes feature table exists'
)
parser.add_argument(
    '--num_obs',
    default=2000,
    help='Number of patients desired in cohort'
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
parser.add_argument(
    '--train_years',
    type=int,
    nargs='+',
    default=[2009, 2010, 2011, 2012, 2013, 2014, 2015],
    help="What years between 2009 and 2021 to use as the training set"
)
parser.add_argument(
    '--test_years',
    type=int,
    nargs='+',
    default=[2016, 2017, 2018, 2019, 2020, 2021],
    help="What years between 2009 and 2021 to use as the test set"
)
parser.add_argument(
    '--evaluate',
    action='store_true',
    help='Whether to evaluate the models on the test set'
)
parser.add_argument(
    '--evaluator',
    default='BinaryEvaluator',
    help='Which type of evaluator to use'
)
parser.add_argument(
    '--config',
    default='Default',
    help='Which feature config to use (changes feature window)'
)

args = parser.parse_args()

# Create dictionary of cohorts, featurizers, feature configs, and trainers
cohort_builders = {
    'InpatientMortalityCohort': InpatientMortalityCohort,
    'LongLengthOfStayCohort': LongLengthOfStayCohort,
    'ThirtyDayReadmission': ThirtyDayReadmission
}
featurizers = {
    'BagOfWordsFeaturizer': BagOfWordsFeaturizer,
}
feature_configs = {
    'Extended': EXTENDED_DEPLOY_CONFIG,
    'Default': DEFAULT_DEPLOY_CONFIG
}
trainers = {
    'LightGBMTrainer': LightGBMTrainer,
}
evaluators = {
    'BinaryEvaluator': BinaryEvaluator,
    'BinaryEvaluatorByTime': BinaryEvaluatorByTime
}

cohort_table_id = f"mining-clinical-decisions.{DATASET_NAME}.{args.experiment_name}_{args.cohort}"
feature_table_id = f"mining-clinical-decisions.{DATASET_NAME}.{args.experiment_name}_{args.cohort}_features"

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

# Add dummy extractor -- needed for case when no feature exists for observation
extractors.append(DummyExtractor(
    cohort_table_id=cohort_table_id,
    feature_table_id=feature_table_id,
))

if args.build_cohort:
    print("Building cohort...")
    c = cohort_builders[args.cohort](
        client=client,
        dataset_name=DATASET_NAME,
        num_obs = args.num_obs,
        table_name=f"{args.experiment_name}_{args.cohort}")
    c()

if args.extract:
    from_table = False
else:
    from_table = True

if args.featurize:
    print("Featurizing...")
    #Set train_years and test_years
    all_years = [i for i in range(2009, 2022)]
    train_years = args.train_years
    test_years = args.test_years
    featurizer = featurizers[args.featurizer](
        cohort_table_id=cohort_table_id,
        feature_table_id=feature_table_id,
        extractors=extractors,
        train_years=train_years,
        test_years=test_years,
        outpath=f"./{args.experiment_name}_{args.cohort}_{args.outpath}",
        feature_config = feature_configs[args.config],
        tfidf=True,
        from_table=from_table
    )
    featurizer()

if args.train:
    print("Training...")
    trainer = trainers[args.trainer](
        working_dir=f"./{args.experiment_name}_{args.cohort}_{args.outpath}"
    )
    trainer(task='label')

if args.evaluate:
    print("Evaluating...")
    evalr = evaluators[args.evaluator](
        outdir = f"./{args.experiment_name}_{args.cohort}_{args.outpath}")

    yhats_path = os.path.join(f"./{args.experiment_name}_{args.cohort}_{args.outpath}",
                            'label_yhats.csv')
    df_test = pd.read_csv(yhats_path)
    if args.evaluator == "BinaryEvaluator":
        evalr(df_test.labels, df_test.predictions)
    elif args.evaluator == "BinaryEvaluatorByTime":
        df_test.index_time = df_test.index_time.astype('datetime64')
        evalr(df_test.labels, df_test.predictions, df_test.index_time)