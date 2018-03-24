#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
. $baseDir/../admin/util.sh


#######################
# variables
#######################
PY=$baseDir/../app/transitionparser/standard.py
TRAIN_DATA=$baseDir/../test/fixtures/standard.recursion.conllu
MODEL=$baseDir/../tmp/standard.fixtures.model
EPOCH=1
LOG_VERBOSITY=1 # info

# functions


# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
set -x
train $PY $LOG_VERBOSITY $MODEL $TRAIN_DATA $EPOCH 
