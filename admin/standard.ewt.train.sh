#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
. $baseDir/util.sh


#######################
# variables
#######################
PY=$baseDir/../app/standard.py
TRAIN_DATA=$baseDir/../data/conll.example
# TRAIN_DATA=$baseDir/../data/UD_English-EWT/en-ud-train.conllu
# MODEL=$baseDir/../tmp/standard.ewt.model
MODEL=$baseDir/../tmp/standard.example.model
EPOCH=1
LOG_VERBOSITY=1 # info

# functions


# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
set -x
train $PY $LOG_VERBOSITY $MODEL $TRAIN_DATA $EPOCH 
