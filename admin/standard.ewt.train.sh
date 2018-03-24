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
PY=$baseDir/../app/transitionparser/standard.py
TRAIN_DATA=$baseDir/../data/UD_English-EWT/en-ud-train.conllu
MODEL=$baseDir/../tmp/standard.ewt.model
EPOCH=1
LOG_VERBOSITY=0 # info

# functions


# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
set -x
train $PY $LOG_VERBOSITY $MODEL $TRAIN_DATA $EPOCH 
