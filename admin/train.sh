#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)

# functions
function train_en(){
    cd $baseDir/../app
    python eager.py \
        --verbosity=0 \
        --epoch=10 \
        --train=True \
        --train_data=$baseDir/../data/UD_English-EWT/en-ud-train.conllu \
        --model=$baseDir/../tmp/eager.en.model
}

function train_zh(){
    cd $baseDir/../app
    python eager.py \
        --verbosity=0 \
        --epoch=10 \
        --train=True \
        --train_data=$baseDir/../data/UD_Chinese-GSD/zh-ud-train.conllu \
        --model=$baseDir/../tmp/eager.zh.model
}

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
set -x
train_zh 