
# healthrex_ml

Tools for creating cohorts, features, and models

1. Clone the repo
```git clone https://github.com/HealthRex/healthrex_ml.git```

2. Create a new environment, note python version
```conda create -n healthrex_ml python=3.7.6```

3. Activate new env
```conda activate healthrex_ml```

3. Install healthrex_ml
```pip install -e .```

4. Install dependencies
```pip install -r requirements.txt```

5. Install lightbm (with conda)
```conda install lightgbm=3.1.1```

##### healthrex_ml/cohorts
Cohort definitions for various supervised ml tasks

##### healthrex_ml/extractors
Grab features from our bq projects

##### healthrex_ml/featurizers
Transform features grabbed by extractors

##### healthrex_ml/models
Define models

##### healthrex_ml/trainers
Train models

##### healthrex_ml/datasets
Pytorch datasets




