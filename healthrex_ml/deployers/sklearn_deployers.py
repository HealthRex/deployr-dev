"""
SklearnDeployr -- defines extractor and inference logic tapping into app orchard
APIs.
"""
import collections
from xml.sax.handler import feature_validation
import pandas as pd
import numpy as np
import os
import xmltodict
import json
import uuid
import requests
from requests.auth import HTTPBasicAuth
from datetime import date, datetime, timedelta
from dateutil.parser import parse
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle
from thefuzz import process
from thefuzz import fuzz

from healthrex_ml.deployers import RACE_MAPPING, VITALSIGN_MAPPING

class NgboostDeployer(object):
    """
    A class that houses attributes and methods necessary to construct a feature
    vector using EPIC and FHIR APIs that plugs into a binary classification
    model trained using an NgboostTrainer. 
    """
    def __init__(self, filepath, credentials, csn, env, client_id, tasks=[], 
                 databricks_endpoint=None, fhir_stu3=None, from_fhir=False):
        """
        Args:   
            filepath: path to deploy config file saved by SklearnTrainer.
                includes: 'model', 'feature_order', 'bin_map', 'feature_config'
            credentials: stores user and password and client id for API calls
            csn: to identify the patient for whom we wish to do inference
            env: prefix to all the EPIC and FHIR apis which also specifies
                the env (ie SUPW, SUPM, POC or Prod)
            client_id: client id for api calls
            fhir_stu3: the fhir_stu3 id of the patient
            from_fhir: if true, must provide fhir_stu3, and inference performed
                using this id instead of csn [useful for debugging]
        """
        self.credentials = credentials
        self.filepath = filepath
        self.csn = csn
        self.client_id = client_id
        self.tasks = tasks
        with open(filepath, 'rb') as f:
            self.deploy = pickle.load(f)
        if databricks_endpoint is None:
            self.clf = self.deploy['model']
        else:
            self.clf = None
        self.databricks_endpoint = databricks_endpoint
        self.feature_types = self.deploy['feature_config']
        self.feature_order = self.deploy['feature_order']
        self.bin_lup = self.deploy['bin_map']
        self.lab_base_names = self.deploy['lab_base_names']
        self.vital_base_names = self.deploy['vital_base_names']
        
        self.api_prefix = env
        self.patient_dict = {'FHIR STU3': fhir_stu3}
        self.from_fhir = from_fhir
        if self.from_fhir:
            assert fhir_stu3 is not None

    def __call__(self, feature_vector=None):
        """
        Get's all necessary features, inputs feature vector into model, returns
        a score. 
        """
        self.get_patient_identifiers()
        if feature_vector is None:
            self.feature_vector = self.populated_features()
            self.feature_vector = np.array(self.feature_vector).reshape(1, -1)
            if self.deploy['transform'] is not None:
                self.feature_vector = self.deploy['transform'].transform(
                    self.feature_vector).toarray()
        else:
            self.feature_vector = feature_vector
        score_dict = {}
        if self.clf is not None:
            for task in self.tasks:
                model, max_iterr = self.clf[task]

                y_mean = model.predict(self.feature_vector, max_iter=max_iterr)
                y_dist = model.pred_dist(self.feature_vector, max_iter=max_iterr).std()

                score_dict[task] = (y_mean[0], y_dist[0])
        else:
            score_dict = self.get_score_from_databricks()
        self.patient_dict['score'] = score_dict
        self.patient_dict = self.get_patient_dict()
        return score_dict

    def get_score_from_databricks(self):
        """
        Sends a HTTPS POST request to a served databricks model that includes
        feature vector in request data. Score is returned as a pandas series
        consistent with how databricks does model serving. Request uses bearer
        authentication. Azure secrets used to store service principal token AND
        the models endpoint.
        """
        def create_tf_serving_json(data):
            return {'inputs': {name: data[name].tolist() for name in 
                               data.keys()} if isinstance(data, dict) else 
                               data.tolist()}
        headers = {
            'Authorization': f'Bearer {os.environ["TDATABRICKS_TOKEN"]}',
            'Content-Type': 'application/json'
        }
        ds_dict = self.feature_vector.to_dict(orient='split') \
            if isinstance(self.feature_vector, pd.DataFrame) \
            else create_tf_serving_json(self.feature_vector)
        data_json = json.dumps(ds_dict, allow_nan=True)
        response = requests.request(method='POST',
                                    headers=headers,
                                    url=self.databricks_endpoint,
                                    data=data_json)
        if response.status_code != 200:
            raise Exception(f'Request failed with status {response.status_code}, {response.text}')
        return response.json()['predictions']

    def compress_feature_vector(self):
        """
        store feature_vector a dict of non zero elements key column inds
        """
        feature_dict = {}
        for i, val in enumerate(self.feature_vector[0]):
            if val != 0:
                if isinstance(val, np.integer):
                    feature_dict[i] = int(val)
                else:
                    feature_dict[i] = val
        return feature_dict

    def get_patient_dict(self):
        """
        Returns dictionary containing score, EPI, and feature_vector
        """
        p_dict = {
            'score': self.patient_dict['score'],
            'model': self.filepath,
            'FHIR STU3': self.patient_dict['FHIR STU3'],
            'FHIR': self.patient_dict['FHIR'],
            'feature_vector': self.compress_feature_vector()
        }
        for key in self.patient_dict:
            if '_error_' in key:
                p_dict[key] = self.patient_dict[key]
        return p_dict

    def get_patient_identifiers(self):
        """
        Populates patient_identifier attribute with EPIC and FHIR identifiers.
        The Rest APIs we'll be using to pull in patient features will require
        different forms of identification for a particular patient.  Here we
        ensure we have all of them. 

        Args:
            patient_csn: CSN at index time for patient in question
        """
        if self.from_fhir:
            id_ = self.patient_dict['FHIR STU3']
            id_type = 'FHIR STU3'
        else:
            id_ = self.csn
            id_type = "SHCMRN" #???
            
        ReqPatientIDs = requests.get(
            (f"{self.api_prefix}api/epic/2010/Common/Patient/"
             "GETPATIENTIDENTIFIERS/Patient/Identifiers"),
            params={'PatientID': id_, 'PatientIDType': id_type},
            headers={'Content-Type': 'application/json; charset=utf-8',
                     'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials["username"],
                               self.credentials["password"])
        )
        patient_json = json.loads(ReqPatientIDs.text)
        for id_load in patient_json["Identifiers"]:
            self.patient_dict[id_load["IDType"]] = id_load["ID"]

    def populated_features(self):
        """
        Extracts a feature vector for the observation ordered according to
        model specifications.
        Returns:
            feature_vector
        """
        if 'Diagnoses' in self.feature_types['Categorical']:
            # print("Featurizing diagnoses")
            self._get_diagnosis_codes()
        if 'PatientProblemGroup' in self.feature_types['Categorical']:
            # print("Featurizing diagnoses")
            self._get_problem_group()

        if 'Medications' in self.feature_types['Categorical']:
            # print("Featurizing medications")
            self._get_medications()
        if 'MedicationGroup' in self.feature_types['Categorical']:
            # print("Featurizing medications")
            self._get_medication_group()
        
        if 'LabResults' in self.feature_types['Numerical']:
            # print("Featurizing lab results")
            self._get_lab_results()
        if 'LabResultsNum' in self.feature_types['Numerical']:
            # print("Featurizing lab results")
            self._get_lab_results_numerical()
        if 'Vitals' in self.feature_types['Numerical']:
            # print("Featurizing vital signs")
            self._get_vital_signs()

        # Handle individual demographic attribute config in helper
        if 'Age' in self.feature_types['Numerical'] or 'Sex' in self.feature_types['Categorical'] or 'Race' in self.feature_types['Categorical']:
            # print('Featurizing demographics')
            self._get_demographics()

        # Create feature vector from patient dictionary
        feature_vector = []
        for feature in self.feature_order:
            if feature in self.patient_dict:
                feature_vector.append(self.patient_dict[feature])
            else:
                feature_vector.append(0)

        return feature_vector

    def _get_problem_group(self):
        look_back = self.feature_types['Categorical']['PatientProblemGroup'][0]['look_back']
        if look_back is None:
            look_back = 50000

        with open('diagnosis_mapping.pkl', 'rb') as f:
            dd_dict = pickle.load(f)
        assert(len(dd_dict) == 72715)

        params = {'patient': self.patient_dict['FHIR'],
                'clinical-status': 'active,inactive,resolved',
                'category': "problem-list-item",
                '_format': 'json'}
        diagnosis_request_response = requests.get(
            f"{self.api_prefix}api/FHIR/R4/Condition",
            params=params,
            headers={'Content-Type': 'application/json; charset=utf-8',
                    'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials['username'],
                            self.credentials['password'])
        ) #MUST CHANGE to self.credentials!!!

        def process_dx_dict(dx_dict):
            for entry in dx_dict['entry']:
                codes = []
                for code in entry['resource']['code']['coding']:
                    if code.get('system', '') == 'http://hl7.org/fhir/sid/icd-10-cm':
                        codes.append(code.get('code'))
                if len(codes) == 0:
                    print(f'Not found any ICD10 code {entry}')
                order_time = entry['resource']['recordedDate'] #['onsetPeriod']['start']??
                try:
                    order_date_time = datetime.strptime(
                        order_time, '%Y-%m-%dT%H:%M:%SZ')
                except:
                    order_date_time = datetime.strptime(order_time, '%Y-%m-%d')

                max_time_delta = timedelta(days=look_back)
                if datetime.now() - order_date_time <= max_time_delta:
                    for code in codes:
                        code = code.replace('.','')
                        if code in dd_dict:
                            dx_group = dd_dict[code]
                            
                            if dx_group in self.patient_dict:
                                self.patient_dict[dx_group] += 1
                            else:
                                self.patient_dict[dx_group] = 1
                        else:
                            print(f'Skip ICD10 {code}')

        
        dx_dict = json.loads(diagnosis_request_response.text)
        process_dx_dict(dx_dict)

    def _get_diagnosis_codes(self):
        """
        Pulls diagnosis codes (ICD10) that were recorded in a patients problem
        list. using EPIC `Condition` API. This is distinct from pulling all
        ICD codes assigned to the patient. 
        
        TODO: figure out overlap between what this returns and what is in our
        clarity extract
        """
        patient_problem_request = requests.get(
            f"{self.api_prefix}api/FHIR/STU3/Condition", #TODO: enable R4 Condition by As
            params={'patient': self.patient_dict['FHIR STU3'],
                    'clinical-status': 'active,inactive,resolved'},
            headers={'Content-Type': 'application/json; charset=utf-8',
                     'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials["username"],
                               self.credentials["password"])
        )
        patient_problem_dict = xmltodict.parse(patient_problem_request.text)
                            
        if 'Bundle' not in patient_problem_dict:  # Some patients restricted
            print('no entries')
            return
        # only one item
        if type(patient_problem_dict['Bundle']['entry']) == \
                collections.OrderedDict:  # otherwise no result
            print("just one entry")
            if "Condition" in patient_problem_dict['Bundle']['entry'][
                    'resource']:
                code = patient_problem_dict['Bundle']['entry']['resource'][
                    'Condition']['code']['coding'][0]['code']['@value']
                self.patient_dict[code] = 1
        else:
            print("many entries")
            for key in tqdm(patient_problem_dict['Bundle']['entry']):
                code = key['resource'][
                    'Condition']['code']['coding'][0]['code']['@value']
                if code not in self.patient_dict:
                    self.patient_dict[code] = 1
                else:
                    self.patient_dict[code] += 1
    def _get_medication_group(self):
        look_back = self.feature_types['Categorical']['MedicationGroup'][0]['look_back']
        with open('rxnorm_to_thera_set.pkl', 'rb') as f:
            rxnorm_to_thera_set = pickle.load(f)


        params = {'patient': self.patient_dict['FHIR'],
            'status': 'active,stopped,completed',
                '_count': '1000',
                '_format': 'json'}
        medication_request_response = requests.get(
            f"{self.api_prefix}api/FHIR/R4/MedicationRequest",
            params=params,
            headers={'Content-Type': 'application/json; charset=utf-8',
                    'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials['username'],
                            self.credentials['password']) #CHANGE!!!
        )
        total_entry = 0
        med_names = {}
        
        def process_med_dict(med_dict, total_entry = 0):
            total_entry = total_entry + len(med_dict['entry'])
            for entry in med_dict['entry']:

                med_name = entry['resource']['medicationReference']['display']
                def get_rxnorm_code(ref):
                    ref = ref[ref.find('Medication')+len('Medication/'):]
                    params = {
                        '_format': 'json'
                    }
                    request_results = requests.get(
                        f"{self.api_prefix}api/FHIR/R4/Medication/{ref}",
                        params=params,
                        headers={'Epic-Client-ID': self.client_id},
                        auth=HTTPBasicAuth(self.credentials['username'],
                                        self.credentials['password'])
                    )
                    med_dict = json.loads(request_results.text)
                    rx_norm_codes = []
                    if 'coding' not in med_dict['code']:
                        #print('RxNorm code not found!')
                        #print((med_dict)
                        return []

                    for c in med_dict['code']['coding']: #c is a dict
                        if c['system'].endswith('rxnorm'):
                            rx_norm_codes.append(c['code'])
                    return rx_norm_codes

                def convert_to_thera_class(rxnorm_codes):
                    results = []
                    for rxnorm in rxnorm_codes:
                        if rxnorm in rxnorm_to_thera_set:
                            results.extend(list(rxnorm_to_thera_set[rxnorm]))
                    return list(set(results))
                rxnorm_codes = get_rxnorm_code(entry['resource']['medicationReference']['reference'])
                thera_classes = convert_to_thera_class(rxnorm_codes)

                #print(med_name, thera_classes)

                order_time = entry['resource']['authoredOn']
                try:
                    order_date_time = datetime.strptime(
                        order_time, '%Y-%m-%dT%H:%M:%SZ')
                except:
                    order_date_time = datetime.strptime(order_time, '%Y-%m-%d')

                max_time_delta = timedelta(days=look_back)
                if datetime.now() - order_date_time <= max_time_delta:
                    for med_name in thera_classes:
                        if med_name in self.patient_dict:
                            self.patient_dict[med_name] += 1
                        else:
                            self.patient_dict[med_name] = 1
            return total_entry

        med_dict = json.loads(medication_request_response.text)
        self.med_dict = med_dict
        process_med_dict(med_dict, total_entry)
        
        

    #NDC, RxNorm, Text <-> med_description, ndc_code, rx_norm_code, limit extractors to FHIR extractable
    def _get_medications(self):
        """
        Pulls patient medications (current and discontinued) from their medical 
        history with using look back window from deploy config file. Populates
        the patient dictionary with the medication names. 

        TODO: this is seemingly only letting me return 1000 entries. They are
        not ordered by time, worried I'm only getting a sample of all existing
        orders during my look back window.  Can I control max entries? 
        """
        params = {
            'patient': self.patient_dict['FHIR'],
        }
        request_results = requests.get(
            f"{self.api_prefix}api/FHIR/R4/MedicationRequest",
            params=params,
            headers={'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials["username"],
                               self.credentials["password"])
        )
        med_dict = xmltodict.parse(request_results.text)

        # No result
        if "Bundle" not in med_dict:
            return

        # One or no results
        if type(med_dict['Bundle']['entry']) == collections.OrderedDict:
            if "MedicationRequest" in med_dict['Bundle']['entry']['resource']:
                med_name = med_dict['Bundle']['entry']['resource'][
                    'MedicationRequest']['medicationReference']['display'][
                    '@value']
                order_time = med_dict['Bundle']['entry']['resource'][
                    'MedicationRequest']['authoredOn']['@value']
                try:
                    order_date_time = datetime.strptime(
                        order_time, '%Y-%m-%dT%H:%M:%SZ')
                except:
                    order_date_time = datetime.strptime(order_time, '%Y-%m-%d')

                look_back = self.feature_types['Categorical']['Medications'][0
                                                                             ]['look_back']
                max_time_delta = timedelta(days=look_back)
                if datetime.now() - order_date_time <= max_time_delta:
                    self.patient_dict[med_name] = 1
            return

        # Multiple results
        for entry in tqdm(med_dict['Bundle']['entry']):
            med_name = entry['resource'][
                'MedicationRequest']['medicationReference']['display']['@value']
            order_time = entry['resource'][
                'MedicationRequest']['authoredOn']['@value']
            try:
                order_date_time = datetime.strptime(
                    order_time, '%Y-%m-%dT%H:%M:%SZ')
            except:
                order_date_time = datetime.strptime(order_time, '%Y-%m-%d')

            look_back = self.feature_types['Categorical']['Medications'][0][
                'look_back']
            max_time_delta = timedelta(days=look_back)
            if datetime.now() - order_date_time <= max_time_delta:
                if med_name in self.patient_dict:
                    self.patient_dict[med_name] += 1
                else:
                    self.patient_dict[med_name] = 1
    def _get_demographics(self):
        """ 
        Calls `GETPATIENTDEMOGRAPHICS` EPIC API and populates patient_dict 
        attribute with demographic information (age, sex, race).
        """
        def calculate_age(born):
            today = date.today()
            return today.year - born.year - ((today.month, today.day)
                                             < (born.month, born.day))

        request_results = requests.get(
            f'{self.api_prefix}api/epic/2019/PatientAccess/Patient/GetPatientDemographics/Patient/Demographics',
            params={'PatientID': self.patient_dict['FHIR'],
                    'PatientIDType': 'FHIR'},
            headers={'Content-Type': 'application/json; charset=utf-8',
                     'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials["username"],
                               self.credentials["password"])
        )

        demographic_json = json.loads(request_results.text)

        # Get binned age
        if 'Age' in self.feature_types['Numerical']:
            self.patient_dict["DOB"] = parse(demographic_json["DateOfBirth"])
            self.patient_dict["Age"] = calculate_age(self.patient_dict["DOB"])
            bin = self._get_bin('Age', self.patient_dict['Age'])
            self.patient_dict[bin] = 1

        #STARR option: Male, Female, Unknown, missing
        if 'Sex' in self.feature_types['Categorical']:
            self.patient_dict[f"sex_{demographic_json['LegalSex']['Title']}"] = 1

        # Race features
        if 'Race' in self.feature_types['Categorical']:
            if 'race' in demographic_json and len(demographic_json['Race'])>=1:
                race_mapped = RACE_MAPPING[demographic_json['Race'][0]['Title']]
                self.patient_dict[f"race_{race_mapped}"] = 1

        ### TODO ###
        # In EPIC multiple races can be listed for a given patient, in STARR
        # data only one race is listed per patient. Here we take the first
        # race listed for a patient to be consistent, but this may not always
        # be correct - probe into how STARR makes this mapping.

    def _get_demographics_old(self):
        """ 
        Calls `GETPATIENTDEMOGRAPHICS` EPIC API and populates patient_dict 
        attribute with demographic information (age, sex, race).
        """
        def calculate_age(born):
            today = date.today()
            return today.year - born.year - ((today.month, today.day)
                                             < (born.month, born.day))

        request_results = requests.get(
            (f'{self.api_prefix}api/epic/2010/Common/Patient/'
             'GETPATIENTDEMOGRAPHICS/Patient/Demographics'),
            params={'PatientID': self.patient_dict['FHIR STU3'],
                    'PatientIDType': 'FHIR STU3'},
            headers={'Content-Type': 'application/json; charset=utf-8',
                     'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials["username"],
                               self.credentials["password"])
        )

        demographic_json = json.loads(request_results.text)

        # Get binned age
        if 'Age' in self.feature_types['Numerical']:
            self.patient_dict["DOB"] = parse(demographic_json["DateOfBirth"])
            self.patient_dict["Age"] = calculate_age(self.patient_dict["DOB"])
            bin = self._get_bin('Age', self.patient_dict['Age'])
            self.patient_dict[bin] = 1

        #STARR option: Male, Female, Unknown, missing
        # Sex features: note EPICs naming convention (gender) is inappropriate.
        if 'Sex' in self.feature_types['Categorical']:
            self.patient_dict[f"sex_{demographic_json['Gender']}"] = 1

        # Race features
        if 'Race' in self.feature_types['Categorical']:
            race_mapped = RACE_MAPPING[demographic_json['Race'].split('^')[0]]
            self.patient_dict[f"race_{race_mapped}"] = 1

        ### TODO ###
        # In EPIC multiple races can be listed for a given patient, in STARR
        # data only one race is listed per patient. Here we take the first
        # race listed for a patient to be consistent, but this may not always
        # be correct - probe into how STARR makes this mapping.
    def _get_lab_results_numerical(self):
        """
        Pulls lab results data for desired base_name component using 
        `GETPATIENTRESULTCOMPONENTS` API. Populates the patient_dict attribute in bag of words fashion.
        """
        look_back = self.feature_types['Numerical']['LabResultsNum'][0]['look_back']
        for base_name in tqdm(self.lab_base_names):
            lab_result_packet = {
                "PatientID": self.patient_dict['FHIR STU3'],
                "PatientIDType": "FHIR STU3",
                "UserID": self.credentials["username"][4:],
                "UserIDType": "External",
                "NumberDaysToLookBack": look_back,
                "MaxNumberOfResults": 200,
                "FromInstant": "",
                "ComponentTypes":
                    [{"Value": base_name, "Type": "base-name"}]
            }
            lab_component_packet = json.dumps(lab_result_packet)
            lab_component_response = requests.post(
                (f'{self.api_prefix}api/epic/2014/Results/Utility/'
                 'GETPATIENTRESULTCOMPONENTS/ResultComponents'),
                headers={
                    'Content-Type': 'application/json; charset=utf-8',
                    'Epic-Client-ID': self.client_id
                },
                auth=HTTPBasicAuth(self.credentials["username"],
                                   self.credentials["password"]),
                data=lab_component_packet
            )
            numeric = '0123456789-.'
            lab_response = json.loads(lab_component_response.text)
            results = []

            if lab_response["ResultComponents"]:  # not none
                for i in range(len(lab_response["ResultComponents"])):
                    if lab_response["ResultComponents"][i]["Value"] is None:
                        continue
                    value = lab_response["ResultComponents"][i]["Value"][0]
                    # Convert to numeric
                    num_value = ''
                    for i, c in enumerate(value):
                        if c in numeric:
                            num_value += c
                    try:
                        value = float(num_value)
                        results.append(value)
                    except:
                        # Log parsing error in patient dictionary
                        self.patient_dict[f"{base_name}_error_{i}"] = value
            if len(results) > 0:
                self.patient_dict[f'NUM__MEAN_{base_name}'] = np.mean(results)
            else:
                self.patient_dict[f'{base_name}_MISSING'] = 1


    def _get_lab_results(self):
        """
        Pulls lab results data for desired base_name component using 
        `GETPATIENTRESULTCOMPONENTS` API. Finds the bins it should be associated
        with and populates the patient_dict attribute in bag of words fashion.

        Ex: if base_name is HCT, and we pull one value of 23.6 corresponding to 
        the 0th bin then self.patient_dict[HCT_0] will be populated with 1.
        """
        look_back = self.feature_types['Numerical']['LabResults'][0
                                                                  ]['look_back']
        for base_name in tqdm(self.lab_base_names):
            lab_result_packet = {
                "PatientID": self.patient_dict['FHIR STU3'],
                "PatientIDType": "FHIR STU3",
                "UserID": self.credentials["username"][4:],
                "UserIDType": "External",
                "NumberDaysToLookBack": look_back,
                "MaxNumberOfResults": 200,
                "FromInstant": "",
                "ComponentTypes":
                    [{"Value": base_name, "Type": "base-name"}]
            }
            lab_component_packet = json.dumps(lab_result_packet)
            lab_component_response = requests.post(
                (f'{self.api_prefix}api/epic/2014/Results/Utility/'
                 'GETPATIENTRESULTCOMPONENTS/ResultComponents'),
                headers={
                    'Content-Type': 'application/json; charset=utf-8',
                    'Epic-Client-ID': self.client_id
                },
                auth=HTTPBasicAuth(self.credentials["username"],
                                   self.credentials["password"]),
                data=lab_component_packet
            )
            numeric = '0123456789-.'
            lab_response = json.loads(lab_component_response.text)
            if lab_response["ResultComponents"]:  # not none
                for i in range(len(lab_response["ResultComponents"])):
                    if lab_response["ResultComponents"][i]["Value"] is None:
                        continue
                    value = lab_response["ResultComponents"][i]["Value"][0]
                    # Convert to numeric
                    num_value = ''
                    for i, c in enumerate(value):
                        if c in numeric:
                            num_value += c
                    try:
                        value = float(num_value)
                        binned_lab_val = self._get_bin(base_name, value)
                        if binned_lab_val not in self.patient_dict:
                            self.patient_dict[binned_lab_val] = 1
                        else:
                            self.patient_dict[binned_lab_val] += 1
                    except:
                        # Log parsing error in patient dictionary
                        self.patient_dict[f"{base_name}_error_{i}"] = value
    

    def _get_vital_signs(self):
        """        
        Gets binned vitals sign features and appends to patient_dict attribute
        in bag of words fashion. Calls the `GETFLOWSHEETROWS` API
        
        Args:
            api_prefix : prefix of the API we're going to call
            look_back : max for `GETFLOWSHEETROWS` is 72 hours
        """
        look_back = self.feature_types['Numerical']['Vitals'][0
                                                                  ]['look_back']
        def get_timestamp_hours_ago(hours):

            # Get the current date and time
            now = datetime.now()

            # Calculate the timedelta for 72 hours (3 days)
            delta = timedelta(hours=hours)

            # Subtract the timedelta from the current date and time
            result = now - delta

            # Format the result as "yyyy-mm-ddThh:mm"
            formatted_result = result.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')

            # Print the formatted result
            return formatted_result

        cutoff_timestamp = get_timestamp_hours_ago(look_back)
        
        params = {'patient': self.patient_dict['FHIR'],
                'category': 'vital-signs',
                'date': f'ge{cutoff_timestamp}',
                '_format': 'json'}
        diagnosis_request_response = requests.get(
            f"{self.api_prefix}api/FHIR/R4/Observation",
            params=params,
            headers={'Content-Type': 'application/json; charset=utf-8',
                    'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials['username'],
                            self.credentials['password'])
        )    
        vital_dict = json.loads(diagnosis_request_response.text)
        vital_signs = {}

        def process_vital_dict(vital_dict):
            for entry in vital_dict['entry']:
                timestamp = entry['resource']['effectiveDateTime']

                try:
                    order_date_time = datetime.strptime(
                        timestamp, '%Y-%m-%dT%H:%M:%SZ')
                except:
                    print(f'Unexpected timestamp format: {timestamp}')
                    order_date_time = datetime.strptime(timestamp, '%Y-%m-%d')

                codes = []
                if 'component' in entry['resource']: #Multiple values included
                    for comp in entry['resource']['component']: #comp is a dict
                        numerical_value = comp['valueQuantity']['value']
                        unit = comp['valueQuantity']['unit']
                        #Sanity check for unit
                        measurement_name = comp['code']['text']

                        if measurement_name in VITALSIGN_MAPPING:
                            measurement_name = VITALSIGN_MAPPING[measurement_name]

                        if measurement_name in vital_signs:
                            vital_signs[measurement_name].append(numerical_value)
                        else:
                            vital_signs[measurement_name] = [numerical_value]
                else:
                    comp = entry['resource']
                    numerical_value = comp['valueQuantity']['value']
                    #unit = comp['valueQuantity']['unit']
                    measurement_name = comp['code']['text']
                    if measurement_name in VITALSIGN_MAPPING:
                        measurement_name = VITALSIGN_MAPPING[measurement_name]
                    if measurement_name in vital_signs:
                        vital_signs[measurement_name].append(numerical_value)
                    else:
                        vital_signs[measurement_name] = [numerical_value]
        process_vital_dict(vital_dict)
        for vital in vital_signs.keys():
            if vital not in list(VITALSIGN_MAPPING.values()) or vital not in list(self.bin_lup['feature']):
                print(f'Skip {vital}')
                continue
            for vital_value in vital_signs[vital]:
                binned_vital_val = self._get_bin(vital, vital_value)
                if binned_vital_val in self.patient_dict:
                    self.patient_dict[binned_vital_val] += 1
                else:
                    self.patient_dict[binned_vital_val] = 1


        

    def _get_bin(self, feature_name, value):
        """
        Given the numerical value for a feature, consults the feature_bin_map
        attribute to find the appropriate bin for said feature and the bag
        of words style name ie `{feature}_{binNumber}`
        
        Args:
            feature_name : name of the numerical feature
            value : floating point associated with value of feature

        Returns:
            feature : `{feature}_{binNumber}` 
                ex: if hematocrit in the 5th bin then HCT_4
        """
        # min_list = (self.bin_lup
        #             .query('feature == @feature_name', engine='python')
        #             ['bin_min'].values
        #             )
        min_list = self.bin_lup[self.bin_lup['feature']
                                == feature_name].values[0]
        min_list = min_list[1:]  # pop first element which is feature name

        for i, m in enumerate(min_list):
            if value < m:
                return f"{feature_name}_{i}"

        return f"{feature_name}_{len(min_list)}"



class SklearnDeployer(object):
    """
    A class that houses attributes and methods necessary to construct a feature
    vector using EPIC and FHIR APIs that plugs into a binary classification
    model trained using an SklearnTrainer. 
    """
    def __init__(self, filepath, credentials, csn, env, client_id,
                 databricks_endpoint=None, fhir_stu3=None, from_fhir=False):
        """
        Args:   
            filepath: path to deploy config file saved by SklearnTrainer.
                includes: 'model', 'feature_order', 'bin_map', 'feature_config'
            credentials: stores user and password and client id for API calls
            csn: to identify the patient for whom we wish to do inference
            env: prefix to all the EPIC and FHIR apis which also specifies
                the env (ie SUPW, SUPM, POC or Prod)
            client_id: client id for api calls
            fhir_stu3: the fhir_stu3 id of the patient
            from_fhir: if true, must provide fhir_stu3, and inference performed
                using this id instead of csn [useful for debugging]
        """
        self.credentials = credentials
        self.filepath = filepath
        self.csn = csn
        self.client_id = client_id
        with open(filepath, 'rb') as f:
            self.deploy = pickle.load(f)
        if databricks_endpoint is None:
            self.clf = self.deploy['model']
        else:
            self.clf = None
        self.databricks_endpoint = databricks_endpoint
        self.feature_types = self.deploy['feature_config']
        self.feature_order = self.deploy['feature_order']
        self.bin_lup = self.deploy['bin_map']
        self.lab_base_names = self.deploy['lab_base_names']
        self.vital_base_names = self.deploy['vital_base_names']
        self.api_prefix = env
        self.patient_dict = {'FHIR STU3': fhir_stu3}
        self.from_fhir = from_fhir
        if self.from_fhir:
            assert fhir_stu3 is not None

    def __call__(self, feature_vector=None):
        """
        Get's all necessary features, inputs feature vector into model, returns
        a score. 
        """
        self.get_patient_identifiers()
        if feature_vector is None:
            self.feature_vector = self.populated_features()
            self.feature_vector = np.array(self.feature_vector).reshape(1, -1)
            if self.deploy['transform'] is not None:
                self.feature_vector = self.deploy['transform'].transform(
                    self.feature_vector).toarray()
        else:
            self.feature_vector = feature_vector
        if self.clf is not None:
            score = self.clf.predict_proba(self.feature_vector)[:, 1][0]
        else:
            score = self.get_score_from_databricks()
        self.patient_dict['score'] = score
        self.patient_dict = self.get_patient_dict()
        return score

    def get_score_from_databricks(self):
        """
        Sends a HTTPS POST request to a served databricks model that includes
        feature vector in request data. Score is returned as a pandas series
        consistent with how databricks does model serving. Request uses bearer
        authentication. Azure secrets used to store service principal token AND
        the models endpoint.
        """
        def create_tf_serving_json(data):
            return {'inputs': {name: data[name].tolist() for name in 
                               data.keys()} if isinstance(data, dict) else 
                               data.tolist()}
        headers = {
            'Authorization': f'Bearer {os.environ["TDATABRICKS_TOKEN"]}',
            'Content-Type': 'application/json'
        }
        ds_dict = self.feature_vector.to_dict(orient='split') \
            if isinstance(self.feature_vector, pd.DataFrame) \
            else create_tf_serving_json(self.feature_vector)
        data_json = json.dumps(ds_dict, allow_nan=True)
        response = requests.request(method='POST',
                                    headers=headers,
                                    url=self.databricks_endpoint,
                                    data=data_json)
        if response.status_code != 200:
            raise Exception(f'Request failed with status {response.status_code}, {response.text}')
        return response.json()['predictions']

    def compress_feature_vector(self):
        """
        store feature_vector a dict of non zero elements key column inds
        """
        feature_dict = {}
        for i, val in enumerate(self.feature_vector[0]):
            if val != 0:
                if isinstance(val, np.integer):
                    feature_dict[i] = int(val)
                else:
                    feature_dict[i] = val
        return feature_dict

    def get_patient_dict(self):
        """
        Returns dictionary containing score, EPI, and feature_vector
        """
        p_dict = {
            'score': self.patient_dict['score'],
            'model': self.filepath,
            'FHIR STU3': self.patient_dict['FHIR STU3'],
            'feature_vector': self.compress_feature_vector()
        }
        for key in self.patient_dict:
            if '_error_' in key:
                p_dict[key] = self.patient_dict[key]
        return p_dict

    def get_patient_identifiers(self):
        """
        Populates patient_identifier attribute with EPIC and FHIR identifiers.
        The Rest APIs we'll be using to pull in patient features will require
        different forms of identification for a particular patient.  Here we
        ensure we have all of them. 

        Args:
            patient_csn: CSN at index time for patient in question
        """
        if self.from_fhir:
            id_ = self.patient_dict['FHIR STU3']
            id_type = 'FHIR STU3'
        else:
            id_ = self.csn
            id_type = "CSN"
        ReqPatientIDs = requests.get(
            (f"{self.api_prefix}api/epic/2010/Common/Patient/"
             "GETPATIENTIDENTIFIERS/Patient/Identifiers"),
            params={'PatientID': id_, 'PatientIDType': id_type},
            headers={'Content-Type': 'application/json; charset=utf-8',
                     'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials["username"],
                               self.credentials["password"])
        )
        patient_json = json.loads(ReqPatientIDs.text)
        for id_load in patient_json["Identifiers"]:
            self.patient_dict[id_load["IDType"]] = id_load["ID"]

    def populated_features(self):
        """
        Extracts a feature vector for the observation ordered according to
        model specifications.
        Returns:
            feature_vector
        """
        if 'Diagnoses' in self.feature_types['Categorical']:
            # print("Featurizing diagnoses")
            self._get_diagnosis_codes()
        if 'Medications' in self.feature_types['Categorical']:
            # print("Featurizing medications")
            self._get_medications()
        if 'LabResults' in self.feature_types['Numerical']:
            # print("Featurizing lab results")
            self._get_lab_results()
        if 'Vitals' in self.feature_types['Numerical']:
            # print("Featurizing vital signs")
            self._get_vital_signs()

        # Handle individual demographic attribute config in helper
        if 'Age' in self.feature_types['Numerical']:
            # print('Featurizing demographics')
            self._get_demographics()

        # Create feature vector from patient dictionary
        feature_vector = []
        for feature in self.feature_order:
            if feature in self.patient_dict:
                feature_vector.append(self.patient_dict[feature])
            else:
                feature_vector.append(0)

        return feature_vector

    def _get_diagnosis_codes(self):
        """
        Pulls diagnosis codes (ICD10) that were recorded in a patients problem
        list. using EPIC `Condition` API. This is distinct from pulling all
        ICD codes assigned to the patient. 
        
        TODO: figure out overlap between what this returns and what is in our
        clarity extract
        """
        patient_problem_request = requests.get(
            f"{self.api_prefix}api/FHIR/STU3/Condition",
            params={'patient': self.patient_dict['FHIR STU3'],
                    'clinical-status': 'active,inactive,resolved'},
            headers={'Content-Type': 'application/json; charset=utf-8',
                     'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials["username"],
                               self.credentials["password"])
        )
        patient_problem_dict = xmltodict.parse(patient_problem_request.text)
                            
        if 'Bundle' not in patient_problem_dict:  # Some patients restricted
            print('no entries')
            return
        # only one item
        if type(patient_problem_dict['Bundle']['entry']) == \
                collections.OrderedDict:  # otherwise no result
            print("just one entry")
            if "Condition" in patient_problem_dict['Bundle']['entry'][
                    'resource']:
                code = patient_problem_dict['Bundle']['entry']['resource'][
                    'Condition']['code']['coding'][0]['code']['@value']
                self.patient_dict[code] = 1
        else:
            print("many entries")
            for key in tqdm(patient_problem_dict['Bundle']['entry']):
                code = key['resource'][
                    'Condition']['code']['coding'][0]['code']['@value']
                if code not in self.patient_dict:
                    self.patient_dict[code] = 1
                else:
                    self.patient_dict[code] += 1

    def _get_medications(self):
        """
        Pulls patient medications (current and discontinued) from their medical 
        history with using look back window from deploy config file. Populates
        the patient dictionary with the medication names. 

        TODO: this is seemingly only letting me return 1000 entries. They are
        not ordered by time, worried I'm only getting a sample of all existing
        orders during my look back window.  Can I control max entries? 
        """
        params = {
            'patient': self.patient_dict['FHIR'],
        }
        request_results = requests.get(
            f"{self.api_prefix}api/FHIR/R4/MedicationRequest",
            params=params,
            headers={'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials["username"],
                               self.credentials["password"])
        )
        med_dict = xmltodict.parse(request_results.text)

        # No result
        if "Bundle" not in med_dict:
            return

        # One or no results
        if type(med_dict['Bundle']['entry']) == collections.OrderedDict:
            if "MedicationRequest" in med_dict['Bundle']['entry']['resource']:
                med_name = med_dict['Bundle']['entry']['resource'][
                    'MedicationRequest']['medicationReference']['display'][
                    '@value']
                order_time = med_dict['Bundle']['entry']['resource'][
                    'MedicationRequest']['authoredOn']['@value']
                try:
                    order_date_time = datetime.strptime(
                        order_time, '%Y-%m-%dT%H:%M:%SZ')
                except:
                    order_date_time = datetime.strptime(order_time, '%Y-%m-%d')

                look_back = self.feature_types['Categorical']['Medications'][0
                                                                             ]['look_back']
                max_time_delta = timedelta(days=look_back)
                if datetime.now() - order_date_time <= max_time_delta:
                    self.patient_dict[med_name] = 1
            return

        # Multiple results
        for entry in tqdm(med_dict['Bundle']['entry']):
            med_name = entry['resource'][
                'MedicationRequest']['medicationReference']['display']['@value']
            order_time = entry['resource'][
                'MedicationRequest']['authoredOn']['@value']
            try:
                order_date_time = datetime.strptime(
                    order_time, '%Y-%m-%dT%H:%M:%SZ')
            except:
                order_date_time = datetime.strptime(order_time, '%Y-%m-%d')

            look_back = self.feature_types['Categorical']['Medications'][0][
                'look_back']
            max_time_delta = timedelta(days=look_back)
            if datetime.now() - order_date_time <= max_time_delta:
                if med_name in self.patient_dict:
                    self.patient_dict[med_name] += 1
                else:
                    self.patient_dict[med_name] = 1
                    
    def _get_medications_group(self, FHIR_ID):
        """
        Pulls patient medications (current and discontinued) from their medical 
        history with using look back window from deploy config file. Populates
        the patient dictionary with the therapy class names. 
        
        TODO: this is seemingly only letting me return 1000 entries. They are
        not ordered by time, worried I'm only getting a sample of all existing
        orders during my look back window.  Can I control max entries? 
        """
        
        def map_medication_to_thera_class(medication_name, mapping_df, matching_thres=50):
            """
            Map medication name to therapy class using fuzzy matching. Mapping_df has medication name as index,  and one column named 'thera_class_abbr'.
            """
            results = process.extract(medication_name.upper(), list(mapping_df.index), limit=1, scorer=fuzz.partial_ratio)
            if results[0][1]>=matching_thres:
                return mapping_df.loc[results[0][0]]['thera_class_abbr']
            else:
                return ''

        params = {
            'patient': FHIR_ID,
        }
        request_results = requests.get(
            f"{os.environ['EPIC_ENV']}api/FHIR/R4/MedicationRequest",
            params=params,
            headers={'Epic-Client-ID': os.environ['EPIC_CLIENT_ID']},
            auth=HTTPBasicAuth(os.environ["secretID"],
                               os.environ["secretpass"])
        )
        med_dict = xmltodict.parse(request_results.text)        
        
        # No result
        if ("Bundle" not in med_dict):
            return
        # One result -> make it a list
        if type(med_dict['Bundle']['entry']) == collections.OrderedDict:
            med_dict['Bundle']['entry'] = [med_dict['Bundle']['entry']]

        #Get look_back window from deployment config
        look_back = self.feature_types['Categorical']['Medications_group'][0][
                'look_back']
        max_time_delta = timedelta(days=look_back)
        # Multiple results
        for entry in tqdm(med_dict['Bundle']['entry']):
            if not("MedicationRequest" in entry['resource']):
                continue
            med_name = entry['resource'][
                'MedicationRequest']['medicationReference']['display']['@value']
            order_time = entry['resource'][
                'MedicationRequest']['authoredOn']['@value']
            try:
                order_date_time = datetime.strptime(
                    order_time, '%Y-%m-%dT%H:%M:%SZ')
            except:
                order_date_time = datetime.strptime(order_time, '%Y-%m-%d')

            if datetime.now() - order_date_time <= max_time_delta:
                thera_class = map_medication_to_thera_class(med_name, self.feature_types['Categorical']['Medications_group'][0]['mapping_df'])
                if thera_class != '':
                    self.patient_dict[thera_class] = 1
                    
    def _get_demographics(self):
        """ 
        Calls `GETPATIENTDEMOGRAPHICS` EPIC API and populates patient_dict 
        attribute with demographic information (age, sex, race).
        """
        def calculate_age(born):
            today = date.today()
            return today.year - born.year - ((today.month, today.day)
                                             < (born.month, born.day))

        request_results = requests.get(
            (f'{self.api_prefix}api/epic/2010/Common/Patient/'
             'GETPATIENTDEMOGRAPHICS/Patient/Demographics'),
            params={'PatientID': self.patient_dict['FHIR STU3'],
                    'PatientIDType': 'FHIR STU3'},
            headers={'Content-Type': 'application/json; charset=utf-8',
                     'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials["username"],
                               self.credentials["password"])
        )

        demographic_json = json.loads(request_results.text)

        # Get binned age
        if 'Age' in self.feature_types['Numerical']:
            self.patient_dict["DOB"] = parse(demographic_json["DateOfBirth"])
            self.patient_dict["Age"] = calculate_age(self.patient_dict["DOB"])
            bin = self._get_bin('Age', self.patient_dict['Age'])
            self.patient_dict[bin] = 1

        # Sex features: note EPICs naming convention (gender) is inappropriate.
        if 'Sex' in self.feature_types['Categorical']:
            self.patient_dict[f"sex_{demographic_json['Gender']}"] = 1

        # Race features
        if 'Race' in self.feature_types['Categorical']:
            race_mapped = RACE_MAPPING[demographic_json['Race'].split('^')[0]]
            self.patient_dict[f"race_{race_mapped}"] = 1

        ### TODO ###
        # In EPIC multiple races can be listed for a given patient, in STARR
        # data only one race is listed per patient. Here we take the first
        # race listed for a patient to be consistent, but this may not always
        # be correct - probe into how STARR makes this mapping.

    def _get_lab_results(self):
        """
        Pulls lab results data for desired base_name component using 
        `GETPATIENTRESULTCOMPONENTS` API. Finds the bins it should be associated
        with and populates the patient_dict attribute in bag of words fashion.

        Ex: if base_name is HCT, and we pull one value of 23.6 corresponding to 
        the 0th bin then self.patient_dict[HCT_0] will be populated with 1.
        """
        look_back = self.feature_types['Numerical']['LabResults'][0
                                                                  ]['look_back']
        for base_name in tqdm(self.lab_base_names):
            lab_result_packet = {
                "PatientID": self.patient_dict['FHIR STU3'],
                "PatientIDType": "FHIR STU3",
                "UserID": self.credentials["username"][4:],
                "UserIDType": "External",
                "NumberDaysToLookBack": look_back,
                "MaxNumberOfResults": 200,
                "FromInstant": "",
                "ComponentTypes":
                    [{"Value": base_name, "Type": "base-name"}]
            }
            lab_component_packet = json.dumps(lab_result_packet)
            lab_component_response = requests.post(
                (f'{self.api_prefix}api/epic/2014/Results/Utility/'
                 'GETPATIENTRESULTCOMPONENTS/ResultComponents'),
                headers={
                    'Content-Type': 'application/json; charset=utf-8',
                    'Epic-Client-ID': self.client_id
                },
                auth=HTTPBasicAuth(self.credentials["username"],
                                   self.credentials["password"]),
                data=lab_component_packet
            )
            numeric = '0123456789-.'
            lab_response = json.loads(lab_component_response.text)
            if lab_response["ResultComponents"]:  # not none
                for i in range(len(lab_response["ResultComponents"])):
                    if lab_response["ResultComponents"][i]["Value"] is None:
                        continue
                    value = lab_response["ResultComponents"][i]["Value"][0]
                    # Convert to numeric
                    num_value = ''
                    for i, c in enumerate(value):
                        if c in numeric:
                            num_value += c
                    try:
                        value = float(num_value)
                        binned_lab_val = self._get_bin(base_name, value)
                        if binned_lab_val not in self.patient_dict:
                            self.patient_dict[binned_lab_val] = 1
                        else:
                            self.patient_dict[binned_lab_val] += 1
                    except:
                        # Log parsing error in patient dictionary
                        self.patient_dict[f"{base_name}_error_{i}"] = value

    def _get_vital_signs(self, api_prefix, look_back=72):
        """
        TODO : Finish implementing this (need it working in SUPD to be able
        to test it)
        ID 5 : BLOOD PRESSURE
        ID 6 : TEMPERATURE
        ID 8 : PULSE
        ID 9 : RESPIRATIONS
        ID 10 : SPO2 (PULSE OXIMETRY)
        
        Gets binned vitals sign features and appends to patient_dict attribute
        in bag of words fashion. Calls the `GETFLOWSHEETROWS` API
        
        Args:
            api_prefix : prefix of the API we're going to call
            look_back : max for `GETFLOWSHEETROWS` is 72 hours
        """

        flowsheet_packet = {
            "PatientID": self.patient_identifiers['EPI'],
            "PatientIDType": "EPI",
            "ContactID": self.patient_identifiers['CSN'],
            "ContactIDType": "CSN",
            "LookbackHours": "72",
            "UserID": "",
            "UserIDType": "",
            "FlowsheetRowIDs": [
                {
                    "ID": "5",
                    "Type": "EXTERNAL"
                }
            ]
        }
        flowsheet_packet = json.dumps(flowsheet_packet)
        flowsheet_response = requests.post(
            (f'{api_prefix}api/epic/2014/Clinical/Patient/'
             'GETFLOWSHEETROWS/FlowsheetRows'),
            headers={'Content-Type': 'application/json; charset=utf-8',
                     'Epic-Client-ID': self.client_id},
            auth=HTTPBasicAuth(self.credentials["username"],
                               self.credentials["password"]),
            data=flowsheet_packet
        )
        flowsheet_data = json.loads(flowsheet_response.text)

        # TODO implement binning and play around with IDS so I can grab a set
        # of flowsheets.

    def _get_bin(self, feature_name, value):
        """
        Given the numerical value for a feature, consults the feature_bin_map
        attribute to find the appropriate bin for said feature and the bag
        of words style name ie `{feature}_{binNumber}`
        
        Args:
            feature_name : name of the numerical feature
            value : floating point associated with value of feature

        Returns:
            feature : `{feature}_{binNumber}` 
                ex: if hematocrit in the 5th bin then HCT_4
        """
        # min_list = (self.bin_lup
        #             .query('feature == @feature_name', engine='python')
        #             ['bin_min'].values
        #             )
        min_list = self.bin_lup[self.bin_lup['feature']
                                == feature_name].values[0]
        min_list = min_list[1:]  # pop first element which is feature name

        for i, m in enumerate(min_list):
            if value < m:
                return f"{feature_name}_{i}"

        return f"{feature_name}_{len(min_list)}"

        # for i in range(len(min_list) - 1):
        #     if i == 0 and value < min_list[i]:  # put in first bin
        #         return f"{feature_name}_{i}"

        #     if value >= min_list[i] and value < min_list[i+1]:
        #         return f"{feature_name}_{i}"

        # Otherwise in last bin
        # return f"{feature_name}_{len(min_list)-1}"
