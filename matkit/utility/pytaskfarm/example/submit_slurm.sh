#!/usr/bin/env bash
#SBATCH --job-name=pytaskfarm
#SBATCH --mail-user=apattama@cityu.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem=2000M
#SBATCH --time=0:30:00
#SBATCH --output=job.out
#SBATCH --error=job.err

export OMP_NUM_THREADS=1
echo $SLURM_JOB_NODELIST
echo $SLURM_CPUS_ON_NODE
echo $SLURM_JOB_NUM_NODES   
scontrol show hostnames $SLURM_JOB_NODELIST
#python pytaskfarm.py --hostfile=$SLURM_JOB_NODELIST --jobfile=jobfile_example --span-nodes=False
