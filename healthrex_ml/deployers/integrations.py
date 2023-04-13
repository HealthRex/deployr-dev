"""
Functions to integrate inferences into EPIC through interconnect.
"""
import json
import os
import requests
from requests.auth import HTTPBasicAuth
import uuid

from healthrex_ml.deployers import EXTERNAL_SCORE_PACKET

import pdb

def write_to_external_score_column(inference, pat_id, score_column_id, 
                                   features=None, contributions=None,
                                   pat_id_type='CSN', POC=True):
    """
    Writes inference, features, and their contirbutions to an external model
    score column using the HANDLEEXTERNALMODELSCORES API

    Args:
        inference: score
        pat_id: patient identifier
        score_column_id: external model identifier (IT creates)
        features: dictionary of (feature <str>, value <float>) pairs
        contributions: dictionary of (feature <str>, contribution <float>) pairs
        pat_id_type: type of identifer (CSN, EXTERNAL, MRN) etc
    """
    # Load parcel
    parcel = EXTERNAL_SCORE_PACKET

    # Configure patient id
    parcel["result"]["results"]["EntityId"][0]["ID"] = pat_id
    parcel["result"]["results"]["EntityId"][0]["Type"] = pat_id_type

    # Configure inference
    parcel["result"]["results"]["Outputs"]["Output1"]["Scores"]["TestScore"][
        "Values"][0] = str(inference)

    # Raw feature values
    if features is not None:
        features = {f : {"Values" : [f"{features[f]}"]} for f in features}
    else: 
        features = {}
    parcel["result"]["results"]["Raw"] = features

    # Feature contribitions
    if contributions is not None:
        contributions = {f: {"Contributions": [f"{contributions[f]}"]} for f in
                         contributions}
    else:
        contributions = {}
    parcel["result"]["results"]["Outputs"]["Output1"][
        "Features"] = contributions

    # Write to Epic
    if POC:
        ENV = os.environ['EPIC_ENV_POC']
        client_id = os.environ['EPIC_CLIENT_ID_POC']
    else:
        ENV = os.environ['EPIC_ENV']
        client_id = os.environ['EPIC_CLIENT_ID']

    parcel = json.dumps(parcel)
    endpoint = (f"{ENV}/api/epic/2017/Reporting/Predictive/"
                f"HANDLEEXTERNALMODELSCORES?modelId=SHC.{score_column_id}@1")
    response = requests.post(
        endpoint,
        params={'jobId': uuid.uuid4()},
        headers={'Content-Type': 'application/json; charset=utf-8'},
        auth=HTTPBasicAuth(os.environ["secretID"], os.environ["secretpass"]), 
        data=parcel
    )
    
def write_smart_data_value(inference, pat_id, sdv_id, pat_id_type='CSN'):
    """
    Write inference to smart data element. Note we can trigger BPA's based
    on what is written to a smart data element. 
    """
    pass
    
