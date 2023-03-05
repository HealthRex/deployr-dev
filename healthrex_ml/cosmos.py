"""
Houses utility functions to read and write to Azure Cosmos
"""
import os

import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
import pandas as pd

def read_predictions_and_labels(container, model, start_date, end_date):
    """
    Grab a batch of json inferences and read in as dataframe
    
    Args:
        container : str specifying container where json items are stored
        model : str specifying which model inference to read
        start_date : grab jsons with inference time >= this date
        end_date : grab jsosn with inference time <= this date
    Returns:
        df : dataframe rep of json items containing model predictions and labels
    """
    client = cosmos_client.CosmosClient(
        os.environ['COSMOS_HOST'],
        os.environ['COSMOS_READ_KEY']
    )
    db = client.get_database_client(os.environ['COSMOS_DB_ID'])
    container = db.get_container_client(container)
    query = f"""
    SELECT 
        *
    FROM 
        c 
    WHERE 
        c.model = '{model}'
    AND
        c.inference_time >= '{start_date}' AND
        c.inference_time <= '{end_date}'
    """

    items = list(container.query_items(
        query=query,
    ))
    df = pd.DataFrame.from_records(items)
    return df

def get_inference_packet(container, date1, date2, partition_key):
    print('\nQuerying for an  Item by Partition Key\n')
    query = f"""
    SELECT 
        c.patient['FHIR STU3'], c.order_date, c.patient.score, c.Error
    FROM 
        c 
    WHERE 
        c.partitionKey = '{partition_key}' AND 
        TimestampToDateTime(c._ts*1000) >= '{date1}' AND 
        TimestampToDateTime(c._ts*1000) <= '{date2}'
    """
    items = list(container.query_items(
        query=query
    ))
    return items

def getcosmoscores(date1, date2, container_id, partition_key):
    client = cosmos_client.CosmosClient(
        os.environ['COSMOS_HOST'],
        os.environ['COSMOS_KEY']
    )
    # setup database for this sample
    try:
        db = client.create_database(id=os.environ['COSMOS_DB_ID'])
        print('Database with id \'{0}\' created'.format(os.environ['COSMOS_DB_ID']))

    except exceptions.CosmosResourceExistsError:
        db = client.get_database_client(os.environ['COSMOS_DB_ID'])
        print('Database with id \'{0}\' was found'.format(os.environ['COSMOS_DB_ID']))

    # setup container for this sample
    try:
        container = db.create_container(id=container_id,
                                        partition_key=PartitionKey(path='/partitionKey'))
        print('Container with id \'{0}\' created'.format(container_id))

    except exceptions.CosmosResourceExistsError:
        container = db.get_container_client(container_id)
        print('Container with id \'{0}\' was found'.format(container_id))

    return get_inference_packet(container, date1, date2, partition_key)
