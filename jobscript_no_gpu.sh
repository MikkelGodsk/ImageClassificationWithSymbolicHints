#!/bin/sh
### General options
### –- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J memory_leak_test
### -- ask for number of cores (default: 1) --
#BSUB -n 1
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --

### -- Starting time hh:mm  (seems to be working) --
##BSUB -b 07:00

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 48:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=64GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o cpu_%J.out
#BSUB -e cpu_%J.err
# -- end of LSF options --

module load python3/3.9.11
module load numpy/1.22.3-python-3.9.11-openblas-0.3.19
source ../torch/bin/activate

python3 compute_class_priors.py