#!/bin/bash

# If no argument is provided, default to nothing
#if [ -z "$RE_CODE" ]; then
#  RE_CODE=""
#fi


set -e  # Exit immediately if a command exits with a non-zero status

# ===VALUE DEFINITION===

PATH_INPUT=$1
PATH_CORPUS=$2
PATH_OUTPUT=$3

#ts_classif :
RELU_PATH=$4
if [ -z "$RELU_PATH" ]; then
    RELU_PATH="base_classif_relu.pt"
fi

TS_SEUIL_FREQ=$5
if [ -z "$TS_SEUIL_FREQ" ]; then
    TS_SEUIL_FREQ=5
fi

TS_BATCH_SIZE=$6
if [ -z "$TS_BATCH_SIZE" ]; then
    TS_BATCH_SIZE=32
fi

TS_VAL_THRESH=$7
if [ -z "$TS_VAL_THRESH" ]; then
    TS_VAL_THRESH=0.143
fi

TS_KEEP_NOISE=$8
if [ -z "$TS_KEEP_NOISE" ]; then
    TS_KEEP_NOISE=False
fi

#de_classif : 
BERT_PATH=$9
if [ -z "$BERT_PATH" ]; then
    BERT_PATH="base_de_bert.pt"
fi

DE_BATCH_SIZE=${10}
if [ -z "$DE_BATCH_SIZE" ]; then
    DE_BATCH_SIZE=32
fi

DE_VAL_THRESH=${11}
if [ -z "$DE_VAL_THRESH" ]; then
    DE_VAL_THRESH=0.5
fi


# ===PIPELINE RUNING===

python ts_classif.py $PATH_INPUT ./results/ts_classif_res.csv --model_path $RELU_PATH --seuil_freq $TS_SEUIL_FREQ --batch_size $TS_BATCH_SIZE --threshold $TS_VAL_THRESH --keep_noise $TS_KEEP_NOISE

sleep 0.1

python de_prestep.py ./results/ts_classif_res.csv $PATH_CORPUS ./results/de_prestep_res.csv

sleep 0.1

python de_classif.py ./results/de_prestep_res.csv ./results/de_classif_res.csv --model_path $BERT_PATH --batch_size $DE_BATCH_SIZE --threshold $DE_VAL_THRESH

sleep 0.1

python de_csv_html.py ./results/de_classif_res.csv $PATH_OUTPUT

