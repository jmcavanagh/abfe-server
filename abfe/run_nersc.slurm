#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -t 48:00:00
#SBATCH -J EASYBFE_ABFE
#SBATCH -o EASYBFE_ABFE.o%j
#SBATCH -A m2834_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH -G 4
#SBATCH --gpu-bind=none
#SBATCH --qos=premium
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ericwangyz@berkeley.edu

source /pscratch/sd/e/eric6/easybfe.env
bash run.sh
