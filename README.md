
# healthrex_ml

Tools for creating cohorts, features, and models


## Installation

### Install a release version

TODO: Publish this to PyPI

This package is intended to be directly hosted on PyPI. Once that happens, just do:
```bash
pip install healthrex_ml
```

Until then, follow the steps below

### Install the latest/local build

This option works well for testing changes to the code locally, and/or for publishing new releases to PyPI.

This repo requires at least Python 3.7.1 and supports up to Python 3.10.8. Steps to build are as-follows:

1. Install Python Poetry here: https://python-poetry.org/
2. Run `poetry build`
3. `pip install` the wheel from the `dist` folder directly, e.g.:
```bash
pip install dist/healthrex_ml-0.1.0-py3-none-any.whl # Whatever is in the `dist` folder
```

To publish to PyPI, run `poetry publish` with the proper credentials.

## Development + Contributing

The quickest way to contribute is to fork the repo and then open a PR against `main`.

Feel free to request access as a collaborator if you expect to update + contribute frequently!

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




