"""
Contains LabelExtactor definitions that pair labels with prospectively 
generated inferences once observable. 
"""
import json
import os
import logging
import requests
from requests.auth import HTTPBasicAuth
import uuid

import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
import pandas as pd
from tqdm import tqdm
import numpy as np

from healthrex_ml.cosmos import getcosmoscores, read_predictions_and_labels
from healthrex_ml.utils import bpa_datetime_parse, result_date_parse


class LabComponentLabelExtractor():

    def __init__(self,
                 base_name,
                 inference_container_id,
                 inference_date_from,
                 inference_date_to,
                 inference_partition_key,
                 output_container_id='predictions_and_labels'):

        self.base_name = base_name
        self.inference_container_id = inference_container_id
        self.inference_partition_key = inference_partition_key
        self.inference_date_from = inference_date_from
        self.inference_date_to = inference_date_to
        self.output_container_id = output_container_id
        self.host = os.environ['COSMOS_HOST']
        self.master_key = os.environ['COSMOS_KEY']
        self.db_id = os.environ['COSMOS_DB_ID']
        self.env = os.environ['EPIC_ENV']
        self.credentials = {
            'username': os.environ['secretID'],
            'password': os.environ['secretpass'],
            'client_id': os.environ['EPIC_CLIENT_ID']
        }

    def __call__(self, test=False):
        self.items = getcosmoscores(self.inference_date_from,
                                    self.inference_date_to,
                                    self.inference_container_id,
                                    self.inference_partition_key)
        self.filter_items_by_passing()
        self.get_labels_and_predictions()
        self.binarize_labels()
        if not test:  # whether to write to cosmos
            self.write_labels_and_predictions()
        else:
            self.write_items = self.labels_to_json()

    def write_labels_and_predictions(self):
        """
        Write labels and predictions to specified container
        """
        client = cosmos_client.CosmosClient(
            self.host, {'masterKey': self.master_key},
            user_agent="CosmosDBPythonQuickstart",
            user_agent_overwrite=True)
        db = client.get_database_client(self.db_id)

        # Get container
        try:
            container = db.create_container(
                id=self.output_container_id,
                partition_key=PartitionKey(path='/model')
            )
        except exceptions.CosmosResourceExistsError:
            container = db.get_container_client(self.output_container_id)

        # Get json items to write
        self.write_items = self.labels_to_json()

        logging.info("Writing inferences to cosmos")
        for item in tqdm(self.write_items):
            container.create_item(body=item)

    def labels_to_json(self):
        """
        Creates list of json objects we will write to cosmos
        """
        labels_and_predictions = []
        for index, row in self.df.iterrows():
            if row['prioritized_time'] != None and not pd.isnull(
                    row['prioritized_time']):
                matching_time = row['prioritized_time'].strftime(
                    '%Y-%m-%d %H:%M:%S')
            else:
                matching_time = None
            item = {
                'id': str(uuid.uuid4()),
                'FHIR_STU3': row['FHIR_STU3'],
                'model': self.inference_partition_key,
                'inference_time': row['inference_time'].strftime(
                    '%Y-%m-%d %H:%M:%S'),
                'result_time': matching_time,
                'label': row['label'],
                'prediction': row['score'],
                'label_string': row['labels_full']
            }
            labels_and_predictions.append(item)
        return labels_and_predictions

    def binarize_labels(self):
        """
        Maps lab abnormality flag to label
        """
        label_mapping = {
            'Abnormal': 1,
            'Normal': 0,
            'Low': 1,
            'Low Panic': 1,
            'Low Off-Scale': 1,
            'High': 1,
            'High Panic': 1,
            'High Off-Scale': 1,
            'Missing': -1  # drop rows with label = -1 for complete case anal
        }
        self.df = self.df.rename(columns={'label': 'labels_full'})
        self.df = self.df.assign(
            label=lambda x: [label_mapping[a] for a in x.labels_full]
        )

    def filter_items_by_passing(self):
        self.passing_items = [i for i in self.items if 'score' in i]
        self.failing_items = [i for i in self.items if 'score' not in i]
        self.frac_failing = len(self.failing_items) / len(self.items)

    def get_labels_and_predictions(self):
        """
        Create a dataframe of inerences and labels
        """
        result_flags = []
        prioritized_times = []
        for item in tqdm(self.passing_items):
            order_time = bpa_datetime_parse(item['order_date'])
            result, closest_time = self.get_closest_lab_result(
                item['FHIR STU3'],
                order_time=order_time
            )
            result_flags.append(result)
            prioritized_times.append(closest_time)

        self.df = pd.DataFrame(data={
            "FHIR_STU3": [it['FHIR STU3'] for it in self.passing_items],
            "inference_time": [bpa_datetime_parse(it['order_date']) for it in
                               self.passing_items],
            "prioritized_time": prioritized_times,
            "score": [it['score'] for it in self.passing_items],
            "label": result_flags
        })

    def get_closest_lab_result(self, fhir_id, order_time):
        """
        Given a dataframe containing rows `fhir_id`, `score`, `order_times`
        find corresponding lab result and reconstruct label. Returns a dataframe
        with `label` columns appended. 
        """
        lab_result_packet = {
            "PatientID": fhir_id,
            "PatientIDType": "FHIR STU3",
            "UserID": self.credentials['username'][4:],
            "UserIDType": "External",
            "NumberDaysToLookBack": 60,
            "FromInstant": "",
            "ComponentTypes":
                [{"Value": self.base_name, "Type": "base-name"}]
        }
        lab_component_packet = json.dumps(lab_result_packet)
        lab_component_response = requests.post(
            (f'{self.env}api/epic/2014/Results/Utility/'
             'GETPATIENTRESULTCOMPONENTS/ResultComponents'),
            headers={
                'Content-Type': 'application/json; charset=utf-8',
                'Epic-Client-ID': self.credentials['client_id']
            },
            auth=HTTPBasicAuth(self.credentials["username"],
                               self.credentials["password"]),
            data=lab_component_packet
        )
        lab_response = json.loads(lab_component_response.text)

        prioritized_times = []
        labels = []
        values = []

        # ResultComp Does not exist
        result_dne = 'ResultComponents' not in lab_response or \
            lab_response['ResultComponents'] is None

        if result_dne:
            return 'Missing', None

        for result in lab_response['ResultComponents']:
            # Time info
            pro_time = f"{result['ResultDate']} {result['ResultTime']}"
            pro_time = result_date_parse(pro_time)
            prioritized_times.append(pro_time)

            # Result info -- consistent with STARR Data - Normal if None
            result_status = result['Abnormality']['Title'] if \
                result['Abnormality'] is not None else 'Normal'
            labels.append(result_status)
            value = result["Value"][0] if result['Value'] is not None else None
            values.append(value)

        df = pd.DataFrame(data={
            'fhir_id': fhir_id,
            'inference_time': order_time,
            'prioritized_times': prioritized_times,
            'labels': labels,
            'values': values
        }
        )
        df = df[df['prioritized_times'] >= order_time]
        if len(df) == 0:
            return "Missing", None
        else:
            df = (df
                  .assign(delta=lambda x: abs(order_time - x.prioritized_times))
                  .sort_values('delta')
                  )
            return df.labels.values[0], df.prioritized_times.values[0]
