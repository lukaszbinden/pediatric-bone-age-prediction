#!/bin/bash

source ~/.bash_profile

#$ -N experiment_yolo_swagger_allin.py
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1

# Array job: -t <range>
#$ -t 1

#$ -l h=node02

#$ -v DISPLAY

#$ -o /data/cvg/lukas/logs/pediatric-bone-age-prediction/experiment_all-in.log

#$ -m ea
#$ -M joel.niklaus@students.unibe.ch,lukas.zbinden@unifr.ch

python experiments/experiment_yolo_swagger_allin.py
