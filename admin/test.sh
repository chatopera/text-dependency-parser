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
    --test=True \
    --model=$baseDir/../tmp/eager.model \
    --test_data=$baseDir/../data/UD_English-EWT/en-ud-test.conllu \
    --test_results=$baseDir/../tmp/en-ud-test.results \