RACE_MAPPING = {
    'White': 'White',
    'Asian': 'Asian',
    'Other': 'Other',
    'Black or African American': 'Black',
    'Declines to State': 'Unknown',
    'Unknown': 'Unknown',
    'Native Hawaiian or Other Pacific Islander': 'Pacific Islander',
    'American Indian or Alaska Native': 'Native American',
    '': 'missing',
}

VITALSIGN_MAPPING = {
    'Systolic blood pressure': 'BP_High_Systolic', 
    'Diastolic blood pressure': 'BP_Low_Diastolic', 
    'Temp': 'Temp', 
    'Pulse': 'Pulse', 
    'Resp Rate': 'Resp', 
    'SpO2': 'SpO2', 
    'Heart Rate': 'Heart Rate'
}


EXTERNAL_SCORE_PACKET = {
    "result": {
        "exit_code": "0",
        "stdout": [
            ""
        ],
        "stderr": [
            ""
        ],
        "results": {
            "EntityId": [
                {
                    "ID": "",
                    "Type": ""
                }
            ],
            "ScoreDisplayed": "TestScore",
            "PredictiveContext": {},
            "OutputType": "5",
            "Outputs": {
                "Output1": {
                    "Scores": {
                        "TestScore": {
                            "Values": ["95"]
                        }
                    },
                    "Features": {
                        "Feature0": {
                            "Contributions": [100]
                        }
                    }
                }
            },
            "Raw": {
                "Feature0": {
                    "Values": ["1"]
                }
            }
        },
        "messages": ["Test message"]
    }

} 
