#!/usr/bin/env bash

DATA_PATH=/shared/hltdir4/disk1/team/data/corpora/co-vax-frames
DATASET=covid19
MISINFO_NAME=co-vax-frames
TRAIN_NAME=co-vax-frames-train
TEST_NAME=co-vax-frames-test

DATASET_PATH=${DATA_PATH}/${DATASET}

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.7.10-0.fc30.x86_64/

mkdir ${DATASET_PATH}/misinfo-index-data

python preprocess/convert_misinfo_to_jsonl.py \
    --input_path ${DATASET_PATH}/${MISINFO_NAME}.json \
    --output_path ${DATASET_PATH}/misinfo-index-data/misinfo.jsonl

python -m pyserini.index \
    -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator \
    -threads 8 \
    -input ${DATASET_PATH}/misinfo-index-data/ \
    -index ${DATASET_PATH}/misinfo-v1 \
    -storePositions \
    -storeDocvectors \
    -storeRaw

python preprocess/search_misinfo_index.py \
    --index_path ${DATASET_PATH}/misinfo-v1 \
    --query_path ${DATASET_PATH}/${TRAIN_NAME}.jsonl \
    --output_path ${DATASET_PATH}/${TRAIN_NAME}-bm25-scores.json \
    --top_k 1000

python preprocess/search_misinfo_index.py \
    --index_path ${DATASET_PATH}/misinfo-v1 \
    --query_path ${DATASET_PATH}/${TEST_NAME}.jsonl \
    --output_path ${DATASET_PATH}/${TEST_NAME}-bm25-scores.json \
    --top_k 1000

python identify/score_predict.py \
  --train_path ${DATASET_PATH}/${TRAIN_NAME}.jsonl \
  --val_path ${DATASET_PATH}/${TEST_NAME}.jsonl \
  --misinfo_path ${DATASET_PATH}/${MISINFO_NAME}.json \
  --threshold_max 10.0 \
  --model_name bm25-scores \
  --train_score_path ${DATASET_PATH}/${TRAIN_NAME}-bm25-scores.json \
  --val_score_path ${DATASET_PATH}/${TEST_NAME}-bm25-scores.json
