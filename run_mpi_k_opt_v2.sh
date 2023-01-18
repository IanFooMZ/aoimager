#!/bin/bash

#SBATCH -A Faraon_Computing
#SBATCH --time=6:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --qos=normal
#SBATCH --mem=100G
#SBATCH --mail-user=ifoo@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source activate mpi-env

module purge
module load mpich/3.3.1

export MPICC=$(which mpicc)

echo $MPICC

BASE_FOLDER=$1

rm -r $BASE_FOLDER
mkdir -p $BASE_FOLDER

python yaml_to_parameters.py $2 $1

srun python optimize_device_hyperparameters.py $BASE_FOLDER/parameters.pickle opt

exit $?
