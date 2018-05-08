#!/bin/bash

#$ -N transfer_learning_winner.py
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1

# Array job: -t <range>
#$ -t 1

#$ -l h=node03

#$ -v DISPLAY

#$ -o transfer_learning_winner.log

#$ -m ea
#$ -M joel.niklaus@students.unibe.ch

python transfer_learning_winner.py