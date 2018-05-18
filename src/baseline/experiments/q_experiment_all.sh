#!/bin/bash

FIRST=$(qsub q_experiment_age_range.sh)
echo $FIRST
SECOND=$(qsub -W depend=afterany:$FIRST q_experiment_classification.sh)
echo $SECOND
THIRD=$(qsub -W depend=afterany:$SECOND q_experiment_disease.sh)
echo $THIRD
