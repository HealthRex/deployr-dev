"""
Base cohort definition
"""

class CohortBuilder():
    """
    A class for constructing a cohort table and saving to a bigquery project.
    A cohort table should have at minimum four columns:
        1. anon_id : the patient id
        2. observation_id : a unique identifier for each observation 
        3. index_time : prediction time for each observation
        4. `label` : binary or multiclass indicating class membership of the
            observations. This may be multiple columns (if in multlilabel
            setting) and will have column names as specified in `label_columns`
            attribute.
    """

    def __init__(self, client, dataset_name, table_name,
                 working_project_id='mining-clinical-decisions'):
        """
        Initializes dataset_name and table_name for where cohort table will be
        saved on bigquery
        """
        self.client = client
        self.project_id = working_project_id
        self.dataset_name = dataset_name
        self.table_name = table_name

    def write_cohort_table(self, overwrite=False, schema=None):
        """
        Writes the cohort dataframe to specified bigquery project, dataset,
        and table with appropriate table schema.

        Args:
            overwrite: if true overwrite existing table
            schema: dictionary of table schema, if None detect automatically. 
        """
        if overwrite:
            if_exists = 'replace'
        else:
            if_exists = 'fail'
        self.df.to_gbq(
            destination_table=f"{self.dataset_name}.{self.table_name}",
            project_id=self.project_id,
            if_exists=if_exists,
            table_schema=schema
        )
