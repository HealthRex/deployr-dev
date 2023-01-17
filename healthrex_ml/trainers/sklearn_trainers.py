"""
Definition of ModelTrainer, SequenceTrainer
"""
from asyncio import Task
from cmath import exp
import os
import json
import pandas as pd
import pickle
import lightgbm as lgb
from lightgbm import early_stopping
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.sparse import load_npz
from tqdm import tqdm

from healthrex_ml.featurizers import DEFAULT_LAB_COMPONENT_IDS
from healthrex_ml.featurizers import DEFAULT_FLOWSHEET_FEATURES

import pdb

class LightGBMTrainer():
    """
    Trains a gbm (LightGBM) and performs appropriate model selection. 
    """

    def __init__(self, working_dir):
        self.working_dir = working_dir

    def __call__(self, task):
        """
        Trains a model against label defined by task
        Args:
            task: column that has label of interest
        """
        self.task = task
        self.clf = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=32
        )

        # Read in train data
        X_train = load_npz(os.path.join(
            self.working_dir, 'train_features.npz'))
        y_train = pd.read_csv(
            os.path.join(self.working_dir, 'train_labels.csv'))

        # Remove any rows with missing labels (for censoring tasks)
        observed_inds = y_train[~y_train[task].isnull()].index
        X_train = X_train[observed_inds]
        y_train = y_train.iloc[observed_inds].reset_index()

        # Create val data
        val_size = int(len(y_train) * 0.15)  # 15 % of training set
        val_inds = y_train.sort_values('index_time', ascending=False).head(
            val_size).index.values
        X_val = X_train[val_inds]
        y_val = y_train.iloc[val_inds]

        # Remove val inds from training set
        train_inds = [idx for idx in y_train.index.values
                      if idx not in val_inds]
        X_train = X_train[train_inds]
        y_train = y_train.iloc[train_inds]

        # Assert val observation ids not in training set
        y_val_obs = set([a for a in y_val.observation_id.values])
        y_train_obs = set([a for a in y_train.observation_id.values])
        assert len(y_val_obs.intersection(y_train_obs)) == 0

        # Read in test data
        X_test = load_npz(os.path.join(self.working_dir, 'test_features.npz'))
        y_test = pd.read_csv(
            os.path.join(self.working_dir, 'test_labels.csv'))

        # Remove censored data from test set
        observed_inds = y_test[~y_test[task].isnull()].index
        X_test = X_test[observed_inds]
        y_test = y_test.iloc[observed_inds].reset_index()

        # Fit model with early stopping
        self.clf.fit(X_train,
                    y_train[self.task].values,
                    eval_set=[(X_val, y_val[self.task].values)],
                    eval_metric=['binary', 'auc'],
                    callbacks=[early_stopping(100)],
                    verbose=1)

        # Predictions
        predictions = self.clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test[self.task], predictions)
        print(f"{self.task} AUC: {round(auc, 2)}")

        df_yhats = pd.DataFrame(data={
            'observation_id' : y_test['observation_id'].values,
            'labels': y_test[self.task].values,
            'predictions': predictions
        })
        yhats_path = f"{self.task}_yhats.csv"
        df_yhats.to_csv(os.path.join(self.working_dir, yhats_path), index=None)

        # Generate config file for DEPLOYR
        self.generate_deploy_config()

    def generate_deploy_config(self):
        """
        Generates the config file used by the deployment module.  Contains
        all information needed for deployment module to create feature vectors
        compatible with the model using EPIC and FHIR APIs. This includes
            1. model: the model itself
            2. feature_order: order of features in feature vector
            3. bin_map: numerical features and min value for each bin
            4. feature_config: dictionary containing which features types used
               in model and their corresponding look back windows. 
            5. tfidf_transform: if tfidf used to transform feature matrix

        """
        deploy = {}
        deploy['model'] = self.clf
        feature_order = pd.read_csv(os.path.join(self.working_dir,
                                                 'feature_order.csv'))
        deploy['feature_order'] = [f for f in feature_order.features]
        if os.path.exists(os.path.join(self.working_dir, 'bin_lup.csv')):
            bin_map = pd.read_csv(os.path.join(self.working_dir, 'bin_lup.csv'),
                                  na_filter=False)
        else:
            bin_map = None
        deploy['bin_map'] = bin_map

        # TFIDF transform
        transform_path = os.path.join(self.working_dir, 'tfidf_transform.pkl')
        if os.path.exists(transform_path):
            with open(transform_path, 'rb') as f:
                transform = pickle.load(f)
        else:
            transform = None
        deploy['transform'] = transform

        with open(os.path.join(self.working_dir, 'feature_config.json'),
                  'r') as f:
            feature_config = json.load(f)
        deploy['feature_config'] = feature_config
        deploy['lab_base_names'] = DEFAULT_LAB_COMPONENT_IDS
        deploy['vital_base_names'] = DEFAULT_FLOWSHEET_FEATURES

        # Dump pickle for DEPLOYR
        with open(os.path.join(self.working_dir, f'{self.task}_deploy.pkl'),
                  'wb') as w:
            pickle.dump(deploy, w)


class BaselineModelTrainer():
    """
    Implements the most basic ML pipeline imagineable. Trains a random forest
    with default hyperparameters using features and labels saved in working
    directory from a Featurizer in featurizers.py

    Generates a deployment config file used in deploy.py that tells us which
    EPIC/FHIR APIs to call, how to construct and order features, and saves
    the model itself. 
    """

    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.task = None  # useful in multilable scenario

    def __call__(self, task):
        """
        Trains a model, saves predictions, saves a config file.
        Args:
            task : column for the binary label
        """
        self.task = task
        self.clf = RandomForestClassifier()
        X_train = load_npz(os.path.join(
            self.working_dir, 'train_features.npz'))
        X_test = load_npz(os.path.join(self.working_dir, 'test_features.npz'))
        y_train = pd.read_csv(
            os.path.join(self.working_dir, 'train_labels.csv'))
        y_test = pd.read_csv(
            os.path.join(self.working_dir, 'test_labels.csv'))

        self.clf.fit(X_train, y_train[self.task])
        predictions = self.clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test[self.task], predictions)
        print(f"AUC: {round(auc, 2)}")

        df_yhats = pd.DataFrame(data={
            'labels': y_test[self.task].values,
            'predictions': predictions
        })
        yhats_path = f"{self.task}_yhats"
        df_yhats.to_csv(os.path.join(self.working_dir, yhats_path), index=None)
        self.generate_deploy_config()

    def generate_deploy_config(self):
        """
        Generates the config file used by the deployment module.  Contains
        all information needed for deployment module to create feature vectors
        compatible with the model using EPIC and FHIR APIs. This includes
            1. model: the model itself
            2. feature_order: order of features in feature vector
            3. bin_map: numerical features and min value for each bin
            4. feature_config: dictionary containing which features types used
               in model and their corresponding look back windows. 
        """
        deploy = {}
        deploy['model'] = self.clf
        feature_order = pd.read_csv(os.path.join(self.working_dir,
                                                 'feature_order.csv'))
        deploy['feature_order'] = [f for f in feature_order.features]
        if os.path.exists(os.path.join(self.working_dir, 'bin_lup.csv')):
            bin_map = pd.read_csv(os.path.join(self.working_dir, 'bin_lup.csv'),
                                  na_filter=False)
        else:
            bin_map = None
        deploy['bin_map'] = bin_map
        with open(os.path.join(self.working_dir, 'feature_config.json'),
                  'r') as f:
            feature_config = json.load(f)
        deploy['feature_config'] = feature_config
        deploy['lab_base_names'] = DEFAULT_LAB_COMPONENT_IDS
        deploy['vital_base_names'] = DEFAULT_FLOWSHEET_FEATURES

        # TFIDF transform
        transform_path = os.path.join(self.working_dir, 'tfidf_transform.pkl')
        if os.path.exists(transform_path):
            with open(transform_path, 'rb') as f:
                transform = pickle.load(f)
        else:
            transform = None
        deploy['transform'] = transform

        with open(os.path.join(self.working_dir, f'{self.task}_deploy.pkl'),
                  'wb') as w:
            pickle.dump(deploy, w)
