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
PY=$baseDir/../app/transitionparser/eager.py
TRAIN_DATA=$baseDir/../data/evsam05/THU/train.conllu
MODEL=$baseDir/../tmp/eager.thu.model
EPOCH=10
LOG_VERBOSITY=0 # info

# functions


# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
set -x
train $PY $LOG_VERBOSITY $MODEL $TRAIN_DATA $EPOCH 
