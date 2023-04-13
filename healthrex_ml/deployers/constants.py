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
