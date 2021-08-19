#!/usr/bin/env bash

DATASET=covid-lies
DATASET_PATH=data/${DATASET}
NUM_SPLITS=5

SPLIT_FILES=""

for (( SPLIT=1; SPLIT<=${NUM_SPLITS}; SPLIT++ )) do
  echo "Training split ${SPLIT} model..."
  MODEL_NAME=covid-twitter-v2-bertscore-covid-lies-scores-s${SPLIT}

  python preprocess/run_bert_score.py \
      --input_path ${DATASET_PATH}/train_s"${SPLIT}".jsonl \
      --misinfo_path ${DATASET_PATH}/misinfo.json \
      --output_path ${DATASET_PATH}/train_s"${SPLIT}"-bert-scores.json \
      --device cuda:0 \
      --batch_size 32 \
      --total_chunks 1

  python preprocess/run_bert_score.py \
      --input_path ${DATASET_PATH}/test_s"${SPLIT}".jsonl \
      --misinfo_path ${DATASET_PATH}/misinfo.json \
      --output_path ${DATASET_PATH}/test_s"${SPLIT}"-bert-scores.json \
      --device cuda:0 \
      --batch_size 32 \
      --total_chunks 1

  python identify/score_predict.py \
    --train_path ${DATASET_PATH}/train_s"${SPLIT}".jsonl \
    --val_path ${DATASET_PATH}/test_s"${SPLIT}".jsonl \
    --misinfo_path ${DATASET_PATH}/misinfo.json \
    --model_name ${MODEL_NAME} \
    --train_score_path ${DATASET_PATH}/train_s"${SPLIT}"-bert-scores.json\
    --val_score_path ${DATASET_PATH}/test_s"${SPLIT}"-bert-scores.json

    SPLIT_FILES="${SPLIT_FILES},models/${MODEL_NAME}/predictions.jsonl"
done

python identify/multi_split_eval.py \
  --input_path ${SPLIT_FILES}
