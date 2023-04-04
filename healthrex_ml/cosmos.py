"""
Houses utility functions to read and write to Azure Cosmos
"""
import datetime
import logging
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


def get_epic_order(patient, partition_key):
    """
    Formats the item to save (appends partition key)
    """
    patient['DOB'] = ''
    order1 = {
        'id': datetime.datetime.now().strftime('%c'),
        'partitionKey': partition_key,
        'orderid': 'None',
        'order_date': datetime.datetime.now().strftime('%c'),
        'patient': patient
    }
    logging.info(f"order1: {order1} End order1")
    return order1


def create_item(container, patient, partition_key):
    """
    Given a container, a patient dictionary, and a partion key, creates the
    item
    Args:
        container: instantiated from cosmoswrite
        patient: dictionary of data to save
        partition_key: string to identify the model that this inference relates
            to
    """
    epic_patient = get_epic_order(patient, partition_key)
    container.create_item(body=epic_patient)


def cosmoswrite(patient, container_id, partition_key):
    """
    Base funtion to call that enables us to write an inference packet
    to cosmos.
    Args:
        patient: a dictionary of items to write to cosmos
        container_id: cosmos container, typically unique to a cohort
        partition_key: a string indicating partition to write to. Partition key
        is unique to a task (perhaps multiple tasks per cohort)
    """
    client = cosmos_client.CosmosClient(
        os.environ['COSMOS_HOST'],
        os.environ['COSMOS_KEY']
    )

    # setup database for this sample
    try:
        db = client.create_database(id=os.environ['COSMOS_DB_ID'])
        print('Database with id \'{0}\' created'.format(
            os.environ['COSMOS_DB_ID']))

    except exceptions.CosmosResourceExistsError:
        db = client.get_database_client(os.environ['COSMOS_DB_ID'])
        print('Database with id \'{0}\' was found'.format(
            os.environ['COSMOS_DB_ID']))

    # setup container for this sample
    try:
        container = db.create_container(id=container_id,
                                        partition_key=PartitionKey(path='/partitionKey'))
        print('Container with id \'{0}\' created'.format(container_id))

    except exceptions.CosmosResourceExistsError:
        container = db.get_container_client(container_id)
        print('Container with id \'{0}\' was found'.format(container_id))

    create_item(container, patient, partition_key)
