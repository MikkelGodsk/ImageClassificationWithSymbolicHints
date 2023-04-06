ImageClassificationWithSymbolicHints
==============================

This repository contains the code for our paper "Image Classification With Symbolic Hints" (*INSERT HYPERLINK*). Here we use a budget-friendly fusion scheme for classifiers which is derived in `Derivation.pdf`.

If you use any of our code, please cite *INSERT CITATION*.

## How to run this project:
- `git clone https://github.com/MikkelGodsk/ImageClassificationWithSymbolicHints.git`
- Fill in `conf/data/data.yml` with URLs to dataset files and the directory to store the dataset in.
- Fill in `DS_DIR` in the makefile to be the folder in which you will store both ImageNet and CMPlaces.
- `make data`
- `make priors`
- `make run_cmplaces` or `make run_imagenet`

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
We use 
- `reliability_diagrams.py` from https://github.com/hollance/reliability-diagrams, and 
- `mapping.json` from https://github.com/DominikFilipiak/imagenet-to-wikidata-mapping (see https://www.tib-op.org/ojs/index.php/bis/article/view/65 for licence)

which are found in `src/third_party_files`.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
