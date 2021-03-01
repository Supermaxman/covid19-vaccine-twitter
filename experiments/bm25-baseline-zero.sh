#!/usr/bin/env bash

DATASET=v2
DATASET_PATH=data/${DATASET}

MISINFO_THRESHOLD_MIN=0.00
MISINFO_THRESHOLD_MAX=5.00
MISINFO_THRESHOLD_STEP=0.0005

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.7.10-0.fc30.x86_64/

mkdir ${DATASET_PATH}/test-misinfo-index-data

python preprocess/convert_misinfo_to_jsonl.py \
    --input_path ${DATASET_PATH}/test_misinfo.json \
    --output_path ${DATASET_PATH}/test-misinfo-index-data/test_misinfo.jsonl

python -m pyserini.index \
    -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator \
    -threads 8 \
    -input ${DATASET_PATH}/test-misinfo-index-data/ \
    -index ${DATASET_PATH}/test-misinfo-v1 \
    -storePositions \
    -storeDocvectors \
    -storeRaw

python preprocess/search_misinfo_index.py \
    --index_path ${DATASET_PATH}/test-misinfo-v1 \
    --query_path ${DATASET_PATH}/train.jsonl \
    --output_path ${DATASET_PATH}/train-bm25-scores.json \
    --top_k 1000

python preprocess/search_misinfo_index.py \
    --index_path ${DATASET_PATH}/test-misinfo-v1 \
    --query_path ${DATASET_PATH}/dev.jsonl \
    --output_path ${DATASET_PATH}/dev-bm25-scores.json \
    --top_k 1000

python preprocess/search_misinfo_index.py \
    --index_path ${DATASET_PATH}/test-misinfo-v1 \
    --query_path ${DATASET_PATH}/test.jsonl \
    --output_path ${DATASET_PATH}/test-bm25-scores.json \
    --top_k 1000

python identify/score_predict.py \
  --train_path ${DATASET_PATH}/train.jsonl \
  --val_path ${DATASET_PATH}/dev.jsonl \
  --misinfo_path ${DATASET_PATH}/test_misinfo.json \
  --model_name bm25-scores-zero \
  --threshold_min ${MISINFO_THRESHOLD_MIN} \
  --threshold_max ${MISINFO_THRESHOLD_MAX} \
  --threshold_step ${MISINFO_THRESHOLD_STEP} \
  --train_score_path ${DATASET_PATH}/train-bm25-scores.json \
  --val_score_path ${DATASET_PATH}/dev-bm25-scores.json

python identify/score_predict.py \
  --train_path ${DATASET_PATH}/train.jsonl \
  --val_path ${DATASET_PATH}/test.jsonl \
  --misinfo_path ${DATASET_PATH}/test_misinfo.json \
  --model_name bm25-scores-zero \
  --threshold_min ${MISINFO_THRESHOLD_MIN} \
  --threshold_max ${MISINFO_THRESHOLD_MAX} \
  --threshold_step ${MISINFO_THRESHOLD_STEP} \
  --train_score_path ${DATASET_PATH}/train-bm25-scores.json \
  --val_score_path ${DATASET_PATH}/test-bm25-scores.json
