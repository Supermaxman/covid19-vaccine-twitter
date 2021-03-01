#!/usr/bin/env bash

DATASET=v2
DATASET_PATH=data/${DATASET}

python preprocess/run_bert_score.py \
    --input_path ${DATASET_PATH}/train.jsonl \
    --misinfo_path ${DATASET_PATH}/train_misinfo.json \
    --output_path ${DATASET_PATH}/train-bert-scores.json \
    --device cuda:0 \
    --batch_size 32

python preprocess/run_bert_score.py \
    --input_path ${DATASET_PATH}/dev.jsonl \
    --misinfo_path ${DATASET_PATH}/dev_misinfo.json \
    --output_path ${DATASET_PATH}/dev-bert-scores.json \
    --device cuda:0 \
    --batch_size 32

python preprocess/run_bert_score.py \
    --input_path ${DATASET_PATH}/test.jsonl \
    --misinfo_path ${DATASET_PATH}/test_misinfo.json \
    --output_path ${DATASET_PATH}/test-bert-scores.json \
    --device cuda:0 \
    --batch_size 32

python identify/score_predict.py \
  --train_path ${DATASET_PATH}/train.jsonl \
  --val_path ${DATASET_PATH}/dev.jsonl \
  --misinfo_path ${DATASET_PATH}/dev_misinfo.json \
  --model_name covid-twitter-v2-bertscore \
  --train_score_path ${DATASET_PATH}/train-bert-scores.json\
  --val_score_path ${DATASET_PATH}/dev-bert-scores.json

python identify/score_predict.py \
  --train_path ${DATASET_PATH}/train.jsonl \
  --val_path ${DATASET_PATH}/test.jsonl \
  --misinfo_path ${DATASET_PATH}/test_misinfo.json \
  --model_name covid-twitter-v2-bertscore \
  --train_score_path ${DATASET_PATH}/train-bert-scores.json\
  --val_score_path ${DATASET_PATH}/test-bert-scores.json
