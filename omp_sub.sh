#!/bin/sh
#An example for serial job.
#SBATCH -J VerT1Mod
#SBATCH -o ver_repo.log
#SBATCH -w node24
#SBATCH -e ver_repo.err
#SBATCH -N 1 -n 34
#SBATCH -p CPU-6248R
echo Running on hosts 
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
ulimit -s unlimited
echo "Begin"
python3  model.py
