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
TRAIN_DATA=$baseDir/../data/UD_English-EWT/en-ud-train.conllu
MODEL=$baseDir/../tmp/eager.ewt.model
EPOCH=200
LOG_VERBOSITY=0 # info

# functions


# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
set -x
cd $baseDir/../app
train $PY $LOG_VERBOSITY $MODEL $TRAIN_DATA $EPOCH 
