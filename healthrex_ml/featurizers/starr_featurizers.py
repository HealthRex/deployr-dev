"""
Definition of BagOfWordsFeaturizer
Definition of SequenceFeaturizer
TODO: Definition of SummaryStatFeaturizer
"""
import json
import os
from re import S
import pandas as pd
import pickle
from google.cloud import bigquery
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfTransformer

from healthrex_ml import extractors
from healthrex_ml.featurizers import DEFAULT_DEPLOY_CONFIG
from healthrex_ml.featurizers import DEFAULT_LAB_COMPONENT_IDS
from healthrex_ml.featurizers import DEFAULT_FLOWSHEET_FEATURES

import pdb

class SequenceFeaturizer():
    """
    Uses FeatureExtractors to generate a set of variable lengh sequences
    for each observation. Saves these sequences as invididual numpy arrays in 
    npz format. The npz contains the following data elements
    NPZ structure:
         sequence: array of array of tokens (continuous values binned), arrays
            grouped by day as in Steinberg et al.
         time_deltas: array with len equal to sequence.shape[0], days
            from index_time for each bag of features
         label: list of labels corresponding to sequence.  len = 1 if one label 
    """

    def __init__(self, cohort_table_id, feature_table_id, train_years,
                 val_years, test_years, label_columns, outpath='./features',
                 project='som-nero-phi-jonc101', dataset='shc_core_2021',
                 feature_config=None):
        """
        Args:
            cohort_table_id: ex 'mining-clinical-decisions.conor_db.table_name'
            feature_table_id: ex
                'mining-clinical-decisions.conor_db.feature_table'
            train_years: list of years to include in training set
            val_years: list of years to include in validation set
            test_years: list of years to include in the test set
            label_columns: list of columns desingated as labels in cohort table
            outpath: path to dump feature matrices
            project: bq project id to extract data from
            dataset: bq dataset with project to extract data from
            feature_config: dictionary with feature types, bins and look back
                windows.
        """
        self.cohort_table_id = cohort_table_id
        self.feature_table_id = feature_table_id
        self.outpath = outpath
        self.project = project
        self.dataset = dataset
        if feature_config is None:
            self.feature_config = DEFAULT_DEPLOY_CONFIG
        else:
            self.feature_config = feature_config
        self.client = bigquery.Client()
        self.train_years = [int(y) for y in train_years]
        self.val_years = [int(y) for y in val_years]
        self.test_years = [int(y) for y in test_years]
        self.label_columns = label_columns

    def __call__(self):
        """
        Generates sequence feature vectors and saves to outpath
        """
        label_cols = ', '.join([f"c.{l}" for l in self.label_columns])
        self.construct_feature_timeline()
        self.collapse_timeline_to_days()
        query = f"""
        SELECT
            f.*, {label_cols}, index_time
        FROM
            {self.feature_table_id}_days f
        INNER JOIN
            {self.cohort_table_id} c
        USING
            (observation_id)
        ORDER BY
            observation_id, time_deltas
        DESC
        """
        df = pd.read_gbq(query, progress_bar_type='tqdm')

        # Split into train, val and test and ensure only terms in train are used
        train_seqs = df[df['index_time'].dt.year.isin(self.train_years)]

        # Build vocab dict and save
        vocab = []
        for feature_list in train_seqs.feature.values:
            if feature_list is not None:
                vocab += [v for v in feature_list.split('---')]
        vocab = set(vocab)
        vocab_map = {}
        counter = 1  # reserve 0 for padding
        for term in vocab:
            vocab_map[term] = counter
            counter += 1

        val_seqs = df[df['index_time'].dt.year.isin(self.val_years)]
        test_seqs = df[df['index_time'].dt.year.isin(self.test_years)]
        seq_dict = {'train': train_seqs, 'val':  val_seqs, 'test': test_seqs}

        # Create working directory if does not already exist and save features
        for dataset, seqs in seq_dict.items():
            os.makedirs(os.path.join(self.outpath, dataset), exist_ok=True)
            print(f"Generating {dataset} sequences")
            for obs in tqdm(seqs.observation_id.unique()):
                example = seqs[seqs['observation_id'] == obs]
                sequence = self.pad_examples(example, vocab_map)
                time_deltas = example.time_deltas.values
                labels = example[self.label_columns].values
                out_file = os.path.join(self.outpath, dataset, f"{obs}.pt")
                torch.save({"sequence": sequence,
                            "time_deltas": time_deltas,
                            "labels": labels}, out_file)

        # Save feature vocab
        with open(os.path.join(self.outpath, 'feature_vocab.npz'), 'w') as fp:
            json.dump(vocab_map, fp)

        # Save bin thresholds if they exist
        self.df_lup = pd.DataFrame()
        for lup in self.lups:
            if lup is not None:
                self.df_lup = pd.concat([self.df_lup, lup])
        if not self.df_lup.empty:
            self.df_lup.to_csv(os.path.join(self.outpath, 'bin_lup.csv'),
                               index=None)

        # Save feature_config
        with open(os.path.join(self.outpath, 'feature_config.json'), 'w') as f:
            json.dump(self.feature_config, f)

    def pad_examples(self, example, vocab):
        """
        Pads days within an example with zeros so all same length. 
        """
        sequences = [torch.tensor(
                    [vocab[a] for a in e.split('---') if a in vocab])
            for e in example.feature.values if e is not None]
        sequences_padded = pad_sequence(sequences, batch_first=True)
        return sequences_padded

    def collapse_timeline_to_days(self):
        """
        Groups long form feature vector by day and collapses all feature values
        """
        query = f"""
        CREATE OR REPLACE TABLE {self.feature_table_id}_days AS (
        SELECT 
            observation_id,
            STRING_AGG(feature, '---') feature,
            TIMESTAMP_DIFF(index_time, feature_time, DAY) time_deltas
        FROM 
            {self.feature_table_id}
        GROUP BY
            observation_id, time_deltas
        )
        """
        query_job = self.client.query(query)
        query_job.result()

    def construct_feature_timeline(self):
        """
        Executes all logic to iteratively append rows to the biq query long form
        feature matrix destination table.  Does this by iteratively joining
        cohort table to tables with desired features, filtering for events
        that occur within each look up range, and then transforming into bag of
        words style representations.  Features with numerical values are binned
        into buckets to enable bag of words repsesentation. 
        """
        fextractors = []
        # Get categorical features
        if 'Sex' in self.feature_config['Categorical']:
            se = extractors.SexExtractor(
                self.cohort_table_id, self.feature_table_id)
            fextractors.append(se)
        if 'Race' in self.feature_config['Categorical']:
            re = extractors.RaceExtractor(self.cohort_table_id,
                                           self.feature_table_id)
            fextractors.append(re)
        if 'Diagnoses' in self.feature_config['Categorical']:
            pe = extractors.PatientProblemExtractor(
                self.cohort_table_id, self.feature_table_id)
            fextractors.append(pe)
        if 'Medications' in self.feature_config['Categorical']:
            me = extractors.MedicationExtractor(
                self.cohort_table_id, self.feature_table_id)
            fextractors.append(me)
        if 'Lab Orders' in self.feature_config['Categorical']:
            lo = extractors.LabOrderExtractor(
                self.cohort_table_id, self.feature_table_id,
                look_back_days=self.feature_config['Categorical'
                    ]['Lab Orders'][0]['look_back'])
            fextractors.append(lo)


        # Get numerical features
        if 'Age' in self.feature_config['Numerical']:
            ae = extractors.AgeExtractor(
                self.cohort_table_id,
                self.feature_table_id,
                bins=self.feature_config['Numerical']['Age'][0]['num_bins'])
            fextractors.append(ae)
        if 'LabResults' in self.feature_config['Numerical']:
            lre = extractors.LabResultBinsExtractor(
                self.cohort_table_id,
                self.feature_table_id,
                base_names=DEFAULT_LAB_COMPONENT_IDS,
                bins=self.feature_config['Numerical']
                ['LabResults'][0]['num_bins'])
            fextractors.append(lre)
        if 'Vitals' in self.feature_config['Numerical']:
            fbe = extractors.FlowsheetBinsExtractor(
                self.cohort_table_id,
                self.feature_table_id,
                flowsheet_descriptions=DEFAULT_FLOWSHEET_FEATURES,
                bins=self.feature_config['Numerical']['Vitals'][0]['num_bins'])
            fextractors.append(fbe)

        # Call extractors and collect any look up tables
        self.lups = []
        for extractor in tqdm(fextractors):
            self.lups.append(extractor())


class BagOfWordsFeaturizer():
    """
    Bag of words but uses FeatureExtractors instead of burying all SQL logic
    as in implementation below. 
    """

    def __init__(self, cohort_table_id, feature_table_id, extractors,
                 train_years=None, test_years=None, outpath='./features',
                 project='som-nero-phi-jonc101', dataset='shc_core_2021',
                 feature_config=None, tfidf=True, from_table=False):
        """
        Args:
            cohort_table_id: ex 'mining-clinical-decisions.conor_db.table_name'
            feature_table_id: ex 
                'mining-clinical-decisions.conor_db.feature_table'
            extractors: list of extractors to pull data from. 
            train_years, test_years : each none by default. If list,
                then specifies which years to include in train, val, test split.
                If none, last year used as test set. 
            outpath: path to dump feature matrices
            project: bq project id to extract data from
            dataset: bq dataset with project to extract data from
            feature_config: dictionary with feature types, bins and look back
                windows. 
            tfidf: if true apply a tfidf transform, train on train apply on
                test
            from_table: default False. If true no feature extraction occurs, 
                sparse matrices created with feature types specified by list of
                extractors 
        """
        self.cohort_table_id = cohort_table_id
        self.feature_table_id = feature_table_id
        self.extractors = extractors
        self.outpath = outpath
        self.project = project
        self.dataset = dataset
        self.tfidf = tfidf
        self.from_table = from_table
        if feature_config is None:
            self.feature_config = DEFAULT_DEPLOY_CONFIG
        else:
            self.feature_config = feature_config
        self.client = bigquery.Client()
        # Get data splits (default last year of data held out as test set)
        split_query = f"""
            SELECT DISTINCT
                EXTRACT(YEAR FROM index_time) year
            FROM
                {self.cohort_table_id}
        """
        df = pd.read_gbq(split_query).sort_values('year')
        if train_years is None:
            self.train_years = df.year.values[:-1]
        else:
            self.train_years = train_years
        if test_years is None:
            self.test_years = [df.year.values[-1]]
        else:
            self.test_years = test_years
        self.replace_table = True

    def __call__(self):
        """
        Executes all logic to construct features and labels and saves all info
        user specified working directory.
        """
        if not self.from_table:
            self.construct_feature_timeline()
            self.construct_bag_of_words_rep()
        else:
            self.lups = []
        feature_types = [f"'{ext.__class__.__name__}'" for ext in self.extractors]
        query = f"""
        SELECT
            *
        FROM
            {self.feature_table_id}_bow
        WHERE
            feature_type in ({','.join(feature_types)})
        ORDER BY
            observation_id
        """
        df = pd.read_gbq(query, progress_bar_type='tqdm')
        train_features = df[df['index_time'].dt.year.isin(self.train_years)]
        apply_features = df[~df['index_time'].dt.year.isin(self.train_years)]
        train_csr, train_ids, train_vocab = self.construct_sparse_matrix(
            train_features, train_features)
        test_csr, test_ids, test_vocab = self.construct_sparse_matrix(
            train_features, apply_features)

        # Apply tfidf transform if indicated and save
        if self.tfidf:
            transform = TfidfTransformer()
            train_csr = transform.fit_transform(train_csr)
            test_csr = transform.transform(test_csr)
            os.makedirs(self.outpath, exist_ok=True)
            transform_path = os.path.join(
                self.outpath, 'tfidf_transform.pkl')
            with open(transform_path, 'wb') as w:
                pickle.dump(transform, w)

        # Query cohort table for labels
        q_cohort = f"""
            SELECT
                *
            FROM
               {self.cohort_table_id}
            ORDER BY
                observation_id
        """
        df_cohort = pd.read_gbq(q_cohort, progress_bar_type='tqdm')
        train_labels = df_cohort[df_cohort['index_time'].dt.year.isin(
            self.train_years)]
        test_labels = df_cohort[~df_cohort['index_time'].dt.year.isin(
            self.train_years)]

        # Sanity check - make sure ids from labels and features in same order
        for a, b in zip(train_labels['observation_id'].values, train_ids):
            try:
                assert a == b
            except:
                pdb.set_trace()
        for a, b in zip(test_labels['observation_id'].values, test_ids):
            assert a == b

        # Create working directory if does not already exist and save features
        os.makedirs(self.outpath, exist_ok=True)
        save_npz(os.path.join(self.outpath, 'train_features.npz'), train_csr)
        save_npz(os.path.join(self.outpath, 'test_features.npz'), test_csr)
        print(f"Feature matrix generated with {train_csr.shape[1]} features")

        # Save labels
        train_labels.to_csv(os.path.join(self.outpath, 'train_labels.csv'),
                            index=None)
        test_labels.to_csv(os.path.join(self.outpath, 'test_labels.csv'),
                           index=None)

        # Save feature order
        df_vocab = pd.DataFrame(data={
            'features': [t for t in train_vocab],
            'indices': [train_vocab[t] for t in train_vocab]
        })
        df_vocab.to_csv(os.path.join(self.outpath, 'feature_order.csv'),
                        index=None)

        # Save bin thresholds if they exist
        self.df_lup = pd.DataFrame()
        for lup in self.lups:
            if lup is not None:
                self.df_lup = pd.concat([self.df_lup, lup])
        if not self.df_lup.empty:
            self.df_lup.to_csv(os.path.join(self.outpath, 'bin_lup.csv'),
                               index=None)

        # Save feature_config
        with open(os.path.join(self.outpath, 'feature_config.json'), 'w') as f:
            json.dump(self.feature_config, f)

    def construct_feature_timeline(self):
        """
        Calls extractors to create long form feature timeline
        """
        # Call extractors and collect any look up tables
        self.lups = []
        for ext in tqdm(self.extractors):
            self.lups.append(ext())

    def construct_bag_of_words_rep(self):
        """
        Transforms long form feature timeline into a bag of words feature
        matrix. Stores as a new table in the given bigquery database but also
        constructs local sparse matrices that can be fed into various sklearn
        style classifiers. 
        """

        # Go from timeline to counts
        query = f"""
        CREATE OR REPLACE TABLE {self.feature_table_id}_bow AS (
        SELECT 
            observation_id, index_time, feature_type, feature, COUNT(*) value 
        FROM 
            {self.feature_table_id}
        WHERE 
            feature_type IS NOT NULL
        AND
            feature IS NOT NULL
        GROUP BY 
            observation_id, index_time, feature_type, feature
        )
        """
        query_job = self.client.query(query)
        query_job.result()

    def construct_sparse_matrix(self, train_features, apply_features):
        """
        Takes long form feature timeline matrix and builds up a scipy csr
        matrix without the costly pivot operation. 
        """
        train_features = (train_features
                          .groupby('observation_id')
                          .agg({'feature': lambda x: list(x),
                                'value': lambda x: list(x)})
                          .reset_index()
                          )
        train_feature_names = [doc for doc in train_features.feature.values]
        train_feature_values = [doc for doc in train_features['value'].values]
        train_obs_id = [id_ for id_ in train_features.observation_id.values]

        apply_features = (apply_features
                          .groupby('observation_id')
                          .agg({'feature': lambda x: list(x),
                                'value': lambda x: list(x)})
                          .reset_index()
                          )
        apply_features_names = [doc for doc in apply_features.feature.values]
        apply_features_values = [doc for doc in apply_features['value'].values]
        apply_obs_id = [id_ for id_ in apply_features.observation_id.values]

        vocabulary = self._build_vocab(train_feature_names)
        indptr = [0]
        indices = []
        data = []
        for i, d in enumerate(apply_features_names):
            for j, term in enumerate(d):
                if term not in vocabulary:
                    continue
                else:
                    indices.append(vocabulary[term])
                    data.append(apply_features_values[i][j])
                if j == 0:
                    # Add zero to data and max index in vocabulary to indices in
                    # case max feature indice isn't in apply features.
                    indices.append(len(vocabulary)-1)
                    data.append(0)
            indptr.append(len(indices))

        csr_data = csr_matrix((data, indices, indptr), dtype=float)

        return csr_data, apply_obs_id, vocabulary

    def _build_vocab(self, data):
        """
        Builds vocabulary of terms from the data. Assigns each unique term
        to a monotonically increasing integer
        """
        vocabulary = {}
        for i, d in enumerate(data):
            for j, term in enumerate(d):
                vocabulary.setdefault(term, len(vocabulary))
        return vocabulary


