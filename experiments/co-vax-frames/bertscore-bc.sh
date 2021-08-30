#!/usr/bin/env bash

DATA_PATH=/shared/hltdir4/disk1/team/data/corpora/co-vax-frames
DATASET=covid19
MISINFO_NAME=co-vax-frames
TRAIN_NAME=co-vax-frames-train
TEST_NAME=co-vax-frames-test

DATASET_PATH=${DATA_PATH}/${DATASET}


python preprocess/run_bert_score.py \
    --input_path ${DATASET_PATH}/${TRAIN_NAME}.jsonl \
    --misinfo_path ${DATASET_PATH}/${MISINFO_NAME}.json \
    --output_path ${DATASET_PATH}/${TRAIN_NAME}-bert-scores.json \
    --device cuda:0 \
    --batch_size 32

python preprocess/run_bert_score.py \
    --input_path ${DATASET_PATH}/${TEST_NAME}.jsonl \
    --misinfo_path ${DATASET_PATH}/${MISINFO_NAME}.json \
    --output_path ${DATASET_PATH}/${TEST_NAME}-bert-scores.json \
    --device cuda:0 \
    --batch_size 32

python identify/score_predict.py \
  --train_path ${DATASET_PATH}/${TRAIN_NAME}.jsonl \
  --val_path ${DATASET_PATH}/${TEST_NAME}.jsonl \
  --misinfo_path ${DATASET_PATH}/${MISINFO_NAME}.json \
  --model_name covid-twitter-v2-bertscore \
  --train_score_path ${DATASET_PATH}/${TRAIN_NAME}-bert-scores.json \
  --val_score_path ${DATASET_PATH}/${TEST_NAME}-bert-scores.json
