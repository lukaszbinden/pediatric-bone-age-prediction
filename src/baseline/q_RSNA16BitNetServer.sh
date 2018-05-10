#!/bin/bash

#$ -N RSNA16BitNetServer.py
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1

# Array job: -t <range>
#$ -t 1

#$ -l h=node03

#$ -v DISPLAY

#$ -o RSNA16BitNetServer.log

#$ -m ea
#$ -M joel.niklaus@students.unibe.ch, lukas.zbinden@unifr.ch

python RSNA16BitNetServer.py

