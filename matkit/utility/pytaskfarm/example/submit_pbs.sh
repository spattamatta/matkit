#!/usr/bin/env bash
#PBS -N htcftmp
#PBS -M apattama@cityu.edu.hk
#PBS -m abe
#PBS -q Area_A
#PBS -l nodes=2:ppn=8
#PBS -l mem=2000M
#PBS -l walltime=11:06:40
#PBS -o job.out
#PBS -e job.err

export OMP_NUM_THREADS=1
cat $PBS_NODEFILE
