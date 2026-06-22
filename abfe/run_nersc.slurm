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
chmod +x run.sh

# This setting will yield a run time around ~3h per-abfe task and roughly ~15-16 abfe runs can be done within 48h
# Each premium queue can harbor 5 jobs, so the throughput should be ~40 abfes/day
./run.sh -l ./v2_1b_1place.sdf -p ../data/9qdz.pdb -c ./config_abfe.yml
# ./run.sh ...
# ./run.sh ...

# if the run is terminated because of the TIMEOUT, you need to use the EAYBFE_FORCE_RUN environment variable to restart the run
# e.g. EAYBFE_FORCE_RUN=1 ./run.sh -l ./v2_1b_1place.sdf -p ../data/9qdz.pdb -c ./config_abfe.yml --force
