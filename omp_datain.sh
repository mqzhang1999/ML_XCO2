#!/bin/sh
#An example for serial job.
#SBATCH -J VerDataT1
#SBATCH -o verDataIn.log
#SBATCH -e verDataIn.err
#SBATCH -N 1 -n 33
#SBATCH -p CPU-6248R
echo Running on hosts 
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
ulimit -s unlimited
echo "Begin"
python3  DataInput.py