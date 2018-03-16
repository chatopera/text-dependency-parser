#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir/../app
python eager.py \
    --verbosity=1 \
    --train=True \
    --train_data=$baseDir/../data/UD_English-EWT/en-ud-train.conllu \
    --model=$baseDir/../tmp/eager.model \