#!/bin/bash

#$ -N experiment_disease.py
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1

# Array job: -t <range>
#$ -t 1

#$ -l h=node03

#$ -v DISPLAY

#$ -o /var/tmp/studi5/boneage/git/jmcs-atml-bone-age-prediction/src/baseline/logs/experiment_disease.log

#$ -m ea
#$ -M joel.niklaus@students.unibe.ch,lukas.zbinden@unifr.ch

python experiments/experiment_disease.py
