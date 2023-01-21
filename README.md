ImageClassificationWithSymbolicHints
==============================

This repository contains the code for the paper "Image Classification With Symbolic Hints" by Jørgensen et al. 2023.

## How to run this project:
- `git clone https://github.com/MikkelGodsk/ImageClassificationWithSymbolicHints.git`
- `conda env create -n ENVNAME --file environment.yml`  (using your own environment name instead of ENVNAME)
- `conda activate ENVNAME`
- get https://github.com/hollance/reliability-diagrams 
- Fill in `conf/data/data.yml`
- `python3 setup.py install`
- `python3 src/data/make_dataset.py`
- `python3 src/experiments/main.py`

## Software used:
Software:
- Python 3.9.11
- cuda 11.3
- cudnn v8.2.0.53-prod-cuda-11.3
- numpy 1.22.3-python-3.9.11-openblas-0.3.19

## Hardware used
The code was run on the main cluster of DTU. Here we used
- NVIDIA-A100 gpus (1 is enough, we used multiple for hyperparameter tuning in parallel). 
- 6 cores of whichever CPU was available, and 64GB ram (mainly for the SVMs).
- 600GB of storage space for the datasets and models.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
