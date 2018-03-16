#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
# functions
function test_en(){
    cd $baseDir/../app
    python eager.py \
        --verbosity=0 \
        --test=True \
        --model=$baseDir/../tmp/eager.en.model \
        --test_data=$baseDir/../data/UD_English-EWT/en-ud-test.conllu \
        --test_results=$baseDir/../tmp/en-ud-test.results
}

function test_zh(){
    cd $baseDir/../app
    python eager.py \
        --verbosity=0 \
        --test=True \
        --model=$baseDir/../tmp/eager.zh.model \
        --test_data=$baseDir/../data/UD_Chinese-GSD/zh-ud-test.conllu \
        --test_results=$baseDir/../tmp/zh-ud-test.results
}


# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
set -x
test_zh