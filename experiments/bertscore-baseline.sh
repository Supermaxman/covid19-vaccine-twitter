#!/usr/bin/env bash

DATASET=v1
DATASET_PATH=data/${DATASET}

python identify/bert_score_baseline.py \
  --train_path ${DATASET_PATH}/train.jsonl \
  --val_path ${DATASET_PATH}/dev.jsonl \
  --misinfo_path ${DATASET_PATH}/misinfo.json \
  --model_name covid-twitter-v2-bertscore \
  --score_path data/scores.json \
  --alt_scores_path data/alternate-scores.json

