"""
Feature extractors that tap into Epic and FHIR APIs
"""
import os
import json
import requests
from requests.auth import HTTPBasicAuth
from pprint import pprint

def fhir_observation_search(fhir_identifier, code=None, category='laboratory'):
    """
    Wrapper around Observeration.Search FHIR R4 API.
    Defaults to searching for all lab results for a given patient.
    Args:
        fhir_identifier (str): patient identifier
        code (str): LOINC code for a specific lab result
            ex: 'http://loinc.org|34487-9'
        category (str): category of observation to search for
    Returns:
        obs_dict (dict): dictionary of observations

    Example:
        >>> fhir_observation_search(os.environ['secretFHIR'],
                                    code='http://loinc.org|34487-9')
    """
    params = {'patient': fhir_identifier,
              'category': category if code is None else None,
              'code': code if code is not None else None,
              '_count': '1000',
              '_format': 'json'}
    observation_request = requests.get(
        f"{os.environ['EPIC_ENV']}api/FHIR/STU3/Observation",
        params=params,
        headers={'Content-Type': 'application/json; charset=utf-8',
                 'Epic-Client-ID': os.environ['EPIC_CLIENT_ID']},
        auth=HTTPBasicAuth(os.environ['secretID'],
                           os.environ['secretpass'])
    )
    obs_dict = json.loads(observation_request.text)
    return obs_dict

# Example usage
# obs_dict = fhir_observation_search(os.environ['secretFHIR'],
#                         code='http://loinc.org|34487-9')
# pprint(obs_dict)