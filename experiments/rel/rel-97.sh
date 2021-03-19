#!/usr/bin/env bash

filename=$(basename -- "$0")
# run names
RUN_ID=${filename::-3}
RUN_NAME=HLTRI_COVID_MISINFO

# collection
DATASET=v1

# major hyper-parameters for system
MISINFO_PRE_MODEL_NAME=digitalepidemiologylab/covid-twitter-bert-v2

MISINFO_THRESHOLD_MIN=0.00
MISINFO_THRESHOLD_MAX=1.00
MISINFO_THRESHOLD_STEP=0.0001
#MISINFO_THRESHOLD=0.995

MISINFO_BATCH_SIZE=6
MISINFO_MAX_SEQ_LEN=96
MISINFO_EMB_SIZE=8
MISINFO_EMB_MODEL=transms
MISINFO_EMB_LOSS_NORM=1
MISINFO_LEARNING_RATE=5e-4
MISINFO_LEARNING_RATE_WARMUP=0.0
MISINFO_WEIGHT_DECAY=0.1
MISINFO_GAMMA=1.0
MISINFO_TRAIN_EPOCHS=20
MISINFO_EVAL_BATCH_SIZE=8

MISINFO_NUM_GPUS=1
MISINFO_TRAIN=true
MISINFO_RUN=false
MISINFO_EVAL=false

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
      --model_name MISINFO-${DATASET}-${RUN_NAME}_${RUN_ID} \
      --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
      --batch_size ${MISINFO_BATCH_SIZE} \
      --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
      --learning_rate ${MISINFO_LEARNING_RATE} \
      --lr_warmup ${MISINFO_LEARNING_RATE_WARMUP} \
      --weight_decay ${MISINFO_WEIGHT_DECAY} \
      --gamma ${MISINFO_GAMMA} \
      --epochs ${MISINFO_TRAIN_EPOCHS} \
      --gpus ${MISINFO_TRAIN_GPUS}
fi

#if [[ ${MISINFO_RUN} = true ]]; then
#    echo "Running dev misinfo..."
#    python identify/predict.py \
#      --model_type ${MISINFO_MODEL_TYPE} \
#      --losses ${MISINFO_LOSSES} \
#      --emb_size ${MISINFO_EMB_SIZE} \
#      --misinfo_path ${DATASET_PATH}/misinfo.json \
#      --val_path ${DATASET_PATH}/dev.jsonl \
#      --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
#      --model_name MISINFO-${DATASET}-${RUN_NAME}_${RUN_ID} \
#      --output_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}_DEV \
#      --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
#      --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
#      --train_sampling ${MISINFO_TRAIN_SAMPLING} \
#      --gpus ${MISINFO_EVAL_GPUS} \
#    ; \
#    python identify/format_predictions.py \
#      --input_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}_DEV \
#      --output_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/dev_scores.json
#
#    echo "Running test misinfo..."
#    python identify/predict.py \
#      --model_type ${MISINFO_MODEL_TYPE} \
#      --losses ${MISINFO_LOSSES} \
#      --emb_size ${MISINFO_EMB_SIZE} \
#      --misinfo_path ${DATASET_PATH}/misinfo.json \
#      --val_path ${DATASET_PATH}/test.jsonl \
#      --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
#      --model_name MISINFO-${DATASET}-${RUN_NAME}_${RUN_ID} \
#      --output_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}_TEST \
#      --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
#      --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
#      --train_sampling ${MISINFO_TRAIN_SAMPLING} \
#      --gpus ${MISINFO_EVAL_GPUS} \
#    ; \
#    python identify/format_predictions.py \
#      --input_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}_TEST \
#      --output_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/test_scores.json
#fi


echo "Freeing ${MISINFO_NUM_GPUS} GPUs: ${MISINFO_GPUS}"
python gpu/free_gpus.py -i ${MISINFO_GPUS}
#
#if [[ ${MISINFO_EVAL} = true ]]; then
#    echo "Evaluating misinfo model..."
#    python identify/score_predict.py \
#      --train_path ${DATASET_PATH}/dev.jsonl \
#      --val_path ${DATASET_PATH}/test.jsonl \
#      --misinfo_path ${DATASET_PATH}/misinfo.json \
#      --model_name MISINFO-${DATASET}-${RUN_NAME}_${RUN_ID} \
#      --train_score_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/dev_scores.json \
#      --val_score_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/test_scores.json \
#      --threshold_min ${MISINFO_THRESHOLD_MIN} \
#      --threshold_max ${MISINFO_THRESHOLD_MAX} \
#      --threshold_step ${MISINFO_THRESHOLD_STEP} \
#      > ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/results.txt \
#      ; \
#      tail -n 1 ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/results.txt
#fi

