#!/usr/bin/env bash

filename=$(basename -- "$0")
# run names
RUN_ID=${filename::-3}
RUN_NAME=HLTRI_COVID_MISINFO

# collection
DATASET=covid-lies
DATASET_PATH=data/${DATASET}
NUM_SPLITS=5

# major hyper-parameters for system
MISINFO_PRE_MODEL_NAME=digitalepidemiologylab/covid-twitter-bert-v2

MISINFO_THRESHOLD_MIN=0.00
MISINFO_THRESHOLD_MAX=1.00
MISINFO_THRESHOLD_STEP=0.0001
#MISINFO_THRESHOLD=0.995

MISINFO_BATCH_SIZE=8
MISINFO_MODEL_TYPE=lm-pairwise
MISINFO_LOSSES=binary_loss
MISINFO_TRAIN_SAMPLING=pairwise
MISINFO_MAX_SEQ_LEN=128
MISINFO_EMB_SIZE=32
MISINFO_LEARNING_RATE=5e-4
MISINFO_TRAIN_EPOCHS=10
MISINFO_EVAL_BATCH_SIZE=8

MISINFO_NUM_GPUS=1
MISINFO_TRAIN=true
MISINFO_RUN=true

export TOKENIZERS_PARALLELISM=true

echo "Starting experiment ${RUN_NAME}_${RUN_ID}"
echo "Reserving ${MISINFO_NUM_GPUS} GPU(s)..."
MISINFO_GPUS=`python gpu/request_gpus.py -r ${MISINFO_NUM_GPUS}`
if [[ ${MISINFO_GPUS} -eq -1 ]]; then
    echo "Unable to reserve ${MISINFO_NUM_GPUS} GPU(s), exiting."
    exit -1
fi
echo "Reserved ${MISINFO_NUM_GPUS} GPUs: ${MISINFO_GPUS}"
MISINFO_TRAIN_GPUS=${MISINFO_GPUS}
MISINFO_EVAL_GPUS=${MISINFO_GPUS}

DATASET_PATH=data/${DATASET}
ARTIFACTS_PATH=artifacts/${DATASET}

# trap ctrl+c to free GPUs
handler()
{
    echo "Experiment aborted."
    echo "Freeing ${MISINFO_NUM_GPUS} GPUs: ${MISINFO_GPUS}"
    python gpu/free_gpus.py -i ${MISINFO_GPUS}
    exit -1
}
trap handler SIGINT


SPLIT_FILES=""
for (( SPLIT=1; SPLIT<=${NUM_SPLITS}; SPLIT++ )) do
  echo "Training split ${SPLIT} model..."
  MODEL_NAME=MISINFO-${DATASET}-${RUN_NAME}_${RUN_ID}-s${SPLIT}

  if [[ ${MISINFO_TRAIN} = true ]]; then
      echo "Training misinfo model..."
      python identify/train.py \
        --model_type ${MISINFO_MODEL_TYPE} \
        --losses ${MISINFO_LOSSES} \
        --emb_size ${MISINFO_EMB_SIZE} \
        --train_misinfo_path ${DATASET_PATH}/misinfo.json \
        --val_misinfo_path ${DATASET_PATH}/misinfo.json \
        --train_path ${DATASET_PATH}/train_s"${SPLIT}".jsonl \
        --val_path ${DATASET_PATH}/dev_s"${SPLIT}".jsonl \
        --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
        --model_name "${MODEL_NAME}" \
        --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
        --batch_size ${MISINFO_BATCH_SIZE} \
        --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
        --train_sampling ${MISINFO_TRAIN_SAMPLING} \
        --learning_rate ${MISINFO_LEARNING_RATE} \
        --epochs ${MISINFO_TRAIN_EPOCHS} \
        --gpus ${MISINFO_TRAIN_GPUS}
  fi

  if [[ ${MISINFO_RUN} = true ]]; then
      echo "Running dev misinfo..."
      python identify/predict.py \
        --model_type ${MISINFO_MODEL_TYPE} \
        --losses ${MISINFO_LOSSES} \
        --emb_size ${MISINFO_EMB_SIZE} \
        --misinfo_path ${DATASET_PATH}/misinfo.json \
        --val_path ${DATASET_PATH}/dev_s"${SPLIT}".jsonl \
        --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
        --model_name "${MODEL_NAME}" \
        --output_path ${ARTIFACTS_PATH}/${MODEL_NAME}_DEV \
        --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
        --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
        --train_sampling ${MISINFO_TRAIN_SAMPLING} \
        --gpus ${MISINFO_EVAL_GPUS} \
      ; \
      python identify/format_predictions.py \
        --input_path ${ARTIFACTS_PATH}/${MODEL_NAME}_DEV \
        --output_path ${ARTIFACTS_PATH}/${MODEL_NAME}/dev_scores.json

      echo "Running test misinfo..."
      python identify/predict.py \
        --model_type ${MISINFO_MODEL_TYPE} \
        --losses ${MISINFO_LOSSES} \
        --emb_size ${MISINFO_EMB_SIZE} \
        --misinfo_path ${DATASET_PATH}/misinfo.json \
        --val_path ${DATASET_PATH}/test_s"${SPLIT}".jsonl \
        --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
        --model_name "${MODEL_NAME}" \
        --output_path ${ARTIFACTS_PATH}/${MODEL_NAME}_TEST \
        --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
        --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
        --train_sampling ${MISINFO_TRAIN_SAMPLING} \
        --gpus ${MISINFO_EVAL_GPUS} \
      ; \
      python identify/format_predictions.py \
        --input_path ${ARTIFACTS_PATH}/${MODEL_NAME}_TEST \
        --output_path ${ARTIFACTS_PATH}/${MODEL_NAME}/test_scores.json
    fi
    echo "Evaluating misinfo model..."
    python identify/score_predict.py \
      --train_path ${DATASET_PATH}/dev_s"${SPLIT}".jsonl \
      --val_path ${DATASET_PATH}/test_s"${SPLIT}".jsonl \
      --misinfo_path ${DATASET_PATH}/misinfo.json \
      --model_name "${MODEL_NAME}" \
      --train_score_path ${ARTIFACTS_PATH}/${MODEL_NAME}/dev_scores.json \
      --val_score_path ${ARTIFACTS_PATH}/${MODEL_NAME}/test_scores.json \
      --threshold_min ${MISINFO_THRESHOLD_MIN} \
      --threshold_max ${MISINFO_THRESHOLD_MAX} \
      --threshold_step ${MISINFO_THRESHOLD_STEP}

    SPLIT_FILES="${SPLIT_FILES},models/${MODEL_NAME}/predictions.jsonl"
done

echo "Freeing ${MISINFO_NUM_GPUS} GPUs: ${MISINFO_GPUS}"
python gpu/free_gpus.py -i ${MISINFO_GPUS}

python identify/multi_split_eval.py \
  --input_path ${SPLIT_FILES}
