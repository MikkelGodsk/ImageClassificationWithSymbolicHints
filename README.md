ImageClassificationWithSymbolicHints
==============================

This repository contains the code for the paper "Image Classification With Symbolic Hints" by Jørgensen et al. 2023.

## How to run this project:
- `git clone https://github.com/MikkelGodsk/ImageClassificationWithSymbolicHints.git`
- `make requirements`
- `conda activate Image_classification_with_symbolic_hints`
- Fill in `conf/data/data.yml` with URLs to dataset files and the directory to store the dataset in.
- `make data`
- `python3 src/experiments/main.py --dataset=<DATASET>` where `<DATASET>` is either `imagenet` or `cmplaces`. 

## Software used:
Software:
- Miniconda 22.11.1
- Python 3.9.11
- cuda 11.3
- cudnn v8.2.0.53-prod-cuda-11.3
- numpy 1.22.3-python-3.9.11-openblas-0.3.19

## Hardware used
The code was run on the main cluster of DTU. Here we used
- NVIDIA-A100 gpus (1 is enough, we used multiple for hyperparameter tuning in parallel). 
- 6 cores of whichever CPU was available, and 64GB ram (mainly for the SVMs).
- 600GB of storage space for the datasets and models.

## Third party files
I use 
- `reliability_diagrams.py` from https://github.com/hollance/reliability-diagrams, and 
- `mapping.json` from https://github.com/DominikFilipiak/imagenet-to-wikidata-mapping (see https://www.tib-op.org/ojs/index.php/bis/article/view/65 for licence)

which are found in `src/third_party_files`.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── conf
        ├── __init__.py    <- Make a configuration module
        ├── data_conf.yaml
        └── data
            └── data.yaml  <- A config file with file names and webpages for downloading the dataset. Should be set by the user.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml    <- The conda environment file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── experiments    <- Scripts with experiments to run
    │   │   └── ...
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── third_party_files
    │       ├── mapping.json <- From https://github.com/DominikFilipiak/imagenet-to-wikidata-mapping
    │       └── reliability_diagrams.py <- From https://github.com/hollance/reliability-diagrams
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
