#!/bin/sh
### General options
### –- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J unpack_imagenet
### -- ask for number of cores (default: 1) --
#BSUB -n 1
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --

### -- Starting time hh:mm  (seems to be working) --
##BSUB -b 07:00

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 48:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s184399@student.dtu.dk
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o cpu_%J.out
#BSUB -e cpu_%J.err
# -- end of LSF options --

for SYNSET in /work3/s184399/ImageNet/UnpackedDataset/train/*.tar; do
    mkdir ${SYNSET%.tar}
    tar -xf $SYNSET -C ${SYNSET%.tar}
done

