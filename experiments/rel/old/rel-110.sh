#!/usr/bin/env bash

filename=$(basename -- "$0")
# run names
RUN_ID=${filename::-3}
RUN_NAME=HLTRI_COVID_MISINFO

# collection
DATASET=v1

# major hyper-parameters for system
MISINFO_PRE_MODEL_NAME=digitalepidemiologylab/covid-twitter-bert-v2

MISINFO_BATCH_SIZE=4
MISINFO_MAX_SEQ_LEN=96
MISINFO_EMB_SIZE=8
MISINFO_EMB_MODEL=transms
MISINFO_EMB_LOSS_NORM=1
MISINFO_LEARNING_RATE=5e-4
MISINFO_LEARNING_RATE_WARMUP=0.1
MISINFO_WEIGHT_DECAY=0.0
MISINFO_GAMMA=1.0
MISINFO_TRAIN_EPOCHS=40
MISINFO_EVAL_BATCH_SIZE=8

MISINFO_NUM_GPUS=4
MISINFO_TRAIN=true
MISINFO_PREDICT=false
MISINFO_EVAL=true

export TOKENIZERS_PARALLELISM=true

echo "Starting experiment ${RUN_NAME}_${RUN_ID}"
echo "Reserving ${MISINFO_NUM_GPUS} GPU(s)..."
MISINFO_GPUS=$(python gpu/request_gpus.py -r ${MISINFO_NUM_GPUS})
if [[ ${MISINFO_GPUS} -eq -1 ]]; then
    echo "Unable to reserve ${MISINFO_NUM_GPUS} GPU(s), exiting."
    exit 1
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
    python gpu/free_gpus.py -i "${MISINFO_GPUS}"
    exit 1
}
trap handler SIGINT

if [[ ${MISINFO_TRAIN} = true ]]; then
    echo "Training misinfo model..."
    python rel/train.py \
      --emb_size ${MISINFO_EMB_SIZE} \
      --emb_model ${MISINFO_EMB_MODEL} \
      --emb_loss_norm ${MISINFO_EMB_LOSS_NORM} \
      --train_misinfo_path ${DATASET_PATH}/misinfo.json \
      --val_misinfo_path ${DATASET_PATH}/misinfo.json \
      --train_path ${DATASET_PATH}/train.jsonl \
      --val_path ${DATASET_PATH}/dev.jsonl \
      --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
      --model_name "MISINFO-${DATASET}-${RUN_NAME}_${RUN_ID}" \
      --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
      --batch_size ${MISINFO_BATCH_SIZE} \
      --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
      --learning_rate ${MISINFO_LEARNING_RATE} \
      --lr_warmup ${MISINFO_LEARNING_RATE_WARMUP} \
      --weight_decay ${MISINFO_WEIGHT_DECAY} \
      --gamma ${MISINFO_GAMMA} \
      --epochs ${MISINFO_TRAIN_EPOCHS} \
      --gpus "${MISINFO_TRAIN_GPUS}"

fi

if [[ ${MISINFO_PREDICT} = true ]]; then
    echo "Running dev misinfo..."
    python rel/predict.py \
      --emb_size ${MISINFO_EMB_SIZE} \
      --emb_model ${MISINFO_EMB_MODEL} \
      --emb_loss_norm ${MISINFO_EMB_LOSS_NORM} \
      --misinfo_path ${DATASET_PATH}/misinfo.json \
      --val_path ${DATASET_PATH}/dev.jsonl \
      --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
      --model_name "MISINFO-${DATASET}-${RUN_NAME}_${RUN_ID}" \
      --output_path "${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}_DEV" \
      --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
      --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
      --gpus "${MISINFO_EVAL_GPUS}"

fi

if [[ ${MISINFO_EVAL} = true ]]; then
    python rel/evaluate.py \
      --emb_size ${MISINFO_EMB_SIZE} \
      --emb_model ${MISINFO_EMB_MODEL} \
      --emb_loss_norm ${MISINFO_EMB_LOSS_NORM} \
      --misinfo_path ${DATASET_PATH}/misinfo.json \
      --val_path ${DATASET_PATH}/dev.jsonl \
      --test_path ${DATASET_PATH}/test.jsonl \
      --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
      --model_name "MISINFO-${DATASET}-${RUN_NAME}_${RUN_ID}" \
      --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
      --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
      --gpus "${MISINFO_EVAL_GPUS}"

fi

echo "Freeing ${MISINFO_NUM_GPUS} GPUs: ${MISINFO_GPUS}"
python gpu/free_gpus.py -i "${MISINFO_GPUS}"

