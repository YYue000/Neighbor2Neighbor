#!/usr/bin/env bash
ROOT=../../../../../..

CONFIG=$1
echo $CONFIG

export PYTHONPATH=$ROOT:$PYTHONPATH 
EVAL_METRICS=bbox

CHECKPOINT=$2
python $ROOT/tools/test.py \
        ${CONFIG} ${CHECKPOINT} --out output/output.pkl --eval $EVAL_METRICS --tmpdir output --gpu-id $3 \
        2>&1|tee test.log
