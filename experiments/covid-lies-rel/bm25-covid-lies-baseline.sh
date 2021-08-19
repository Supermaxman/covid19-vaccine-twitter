#!/usr/bin/env bash

DATASET=covid-lies
DATASET_PATH=data/${DATASET}
NUM_SPLITS=5

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.7.10-0.fc30.x86_64/

#mkdir ${DATASET_PATH}/misinfo-index-data
#
#python preprocess/convert_misinfo_to_jsonl.py \
#    --input_path ${DATASET_PATH}/misinfo.json \
#    --output_path ${DATASET_PATH}/misinfo-index-data/misinfo.jsonl
#
#python -m pyserini.index \
#    -collection JsonCollection \
#    -generator DefaultLuceneDocumentGenerator \
#    -threads 8 \
#    -input ${DATASET_PATH}/misinfo-index-data/ \
#    -index ${DATASET_PATH}/misinfo-v1 \
#    -storePositions \
#    -storeDocvectors \
#    -storeRaw

for (( SPLIT=1; SPLIT<=${NUM_SPLITS}; SPLIT++ )) do
  echo "Training split ${SPLIT} model..."
  MODEL_NAME=bm25-covid-lies-scores-s${SPLIT}

  python preprocess/search_misinfo_index.py \
      --index_path ${DATASET_PATH}/misinfo-v1 \
      --query_path ${DATASET_PATH}/train_s"${SPLIT}".jsonl \
      --output_path ${DATASET_PATH}/train_s"${SPLIT}"-bm25-scores.json \
      --top_k 1000

  python preprocess/search_misinfo_index.py \
      --index_path ${DATASET_PATH}/misinfo-v1 \
      --query_path ${DATASET_PATH}/dev_s"${SPLIT}".jsonl \
      --output_path ${DATASET_PATH}/dev_s"${SPLIT}"-bm25-scores.json \
      --top_k 1000

  python preprocess/search_misinfo_index.py \
      --index_path ${DATASET_PATH}/misinfo-v1 \
      --query_path ${DATASET_PATH}/test_s"${SPLIT}".jsonl \
      --output_path ${DATASET_PATH}/test_s"${SPLIT}"-bm25-scores.json \
      --top_k 1000

  python identify/score_predict.py \
    --train_path ${DATASET_PATH}/train_s"${SPLIT}".jsonl \
    --val_path ${DATASET_PATH}/dev_s"${SPLIT}".jsonl \
    --misinfo_path ${DATASET_PATH}/misinfo.json \
    --model_name ${MODEL_NAME} \
    --train_score_path ${DATASET_PATH}/train_s"${SPLIT}"-bm25-scores.json \
    --val_score_path ${DATASET_PATH}/dev_s"${SPLIT}"-bm25-scores.json

  python identify/score_predict.py \
    --train_path ${DATASET_PATH}/train_s"${SPLIT}".jsonl \
    --val_path ${DATASET_PATH}/test_s"${SPLIT}".jsonl \
    --misinfo_path ${DATASET_PATH}/misinfo.json \
    --threshold_max 10.0 \
    --model_name ${MODEL_NAME} \
    --train_score_path ${DATASET_PATH}/train_s"${SPLIT}"-bm25-scores.json \
    --val_score_path ${DATASET_PATH}/test_s"${SPLIT}"-bm25-scores.json
done
