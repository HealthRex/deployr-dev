"""
Test integration of deployers/integrations.py with Epic's HANDLEEXTERNALMODELSCORES API
"""

import os

from healthrex_ml.deployers import write_to_external_score_column

def test_write_to_external_score_column():
    """
    Test write_to_external_score_column
    """
    # Configure
    inference = 51 # score betweeen 0 and 100 (configurable in Epic)
    pat_id = os.environ['TEST_CSN']
    score_column_id = os.environ['TEST_SCORE_COLUMN_ID']
    features = {'feature1' : 1.0, 'feature2' : 2.0}
    contributions = {'feature1' : 0.5, 'feature2' : 0.5}
    pat_id_type = 'CSN'
    POC = True

    # Write to Epic
    write_to_external_score_column(
        inference, pat_id, score_column_id, 
        features=features, contributions=contributions,
        pat_id_type=pat_id_type, POC=POC)

if __name__ == '__main__':
    test_write_to_external_score_column()
    