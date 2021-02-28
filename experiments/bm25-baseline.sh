#!/usr/bin/env bash

DATASET=v1
DATASET_PATH=data/${DATASET}

#python identify/score_baseline.py \
#  --train_path ${DATASET_PATH}/train.jsonl \
#  --val_path ${DATASET_PATH}/dev.jsonl \
#  --misinfo_path ${DATASET_PATH}/misinfo.json \
#  --model_name covid-twitter-v2-bertscore \
#  --score_path data/scores.json

mkdir data/misinfo-index-data

python preprocess/convert_misinfo_to_jsonl.py \
    --input_path ${DATASET_PATH}/misinfo.json \
    --output_path ${DATASET_PATH}/misinfo.jsonl

python -m pyserini.index \
    -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator \
    -threads 8 \
    -input ${DATASET_PATH}/misinfo.jsonl \
    -index ${DATASET_PATH}/misinfo-v1 \
    -storePositions \
    -storeDocvectors \
    -storeRaw

python preprocess/search_misinfo_index.py \
    --index_path ${DATASET_PATH}/misinfo-v1 \
    --query_path ${DATASET_PATH}/train.jsonl \
    --output_path ${DATASET_PATH}/train-bm25-scores.json \
    --top_k 1000

python preprocess/search_misinfo_index.py \
    --index_path ${DATASET_PATH}/misinfo-v1 \
    --query_path ${DATASET_PATH}/dev.jsonl \
    --output_path ${DATASET_PATH}/dev-bm25-scores.json \
    --top_k 1000

python preprocess/search_misinfo_index.py \
    --index_path ${DATASET_PATH}/misinfo-v1 \
    --query_path ${DATASET_PATH}/test.jsonl \
    --output_path ${DATASET_PATH}/test-bm25-scores.json \
    --top_k 1000

python identify/score_baseline.py \
  --train_path ${DATASET_PATH}/train.jsonl \
  --val_path ${DATASET_PATH}/dev.jsonl \
  --misinfo_path ${DATASET_PATH}/misinfo.json \
  --model_name bm25-scores \
  --train_score_path ${DATASET_PATH}/train-bm25-scores.json \
  --val_score_path ${DATASET_PATH}/dev-bm25-scores.json

python identify/score_baseline.py \
  --train_path ${DATASET_PATH}/train.jsonl \
  --val_path ${DATASET_PATH}/test.jsonl \
  --misinfo_path ${DATASET_PATH}/misinfo.json \
  --model_name bm25-scores \
  --train_score_path ${DATASET_PATH}/train-bm25-scores.json \
  --val_score_path ${DATASET_PATH}/test-bm25-scores.json