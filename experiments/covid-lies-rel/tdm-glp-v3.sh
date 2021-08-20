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

MISINFO_BATCH_SIZE=6
MISINFO_MAX_SEQ_LEN=96
MISINFO_EMB_SIZE=8
MISINFO_EMB_MODEL=transms
MISINFO_EMB_LOSS_NORM=1
MISINFO_LEARNING_RATE=5e-4
MISINFO_GAMMA=1.0
MISINFO_TRAIN_EPOCHS=10
MISINFO_EVAL_BATCH_SIZE=8
MISINFO_EVAL_MODE=centroid

MISINFO_NUM_GPUS=1
MISINFO_TRAIN=false
MISINFO_RUN=false

export TOKENIZERS_PARALLELISM=true

echo "Starting experiment ${RUN_NAME}_${RUN_ID}"
echo "Reserving ${MISINFO_NUM_GPUS} GPU(s)..."
#MISINFO_GPUS=`python gpu/request_gpus.py -r ${MISINFO_NUM_GPUS}`
#if [[ ${MISINFO_GPUS} -eq -1 ]]; then
#    echo "Unable to reserve ${MISINFO_NUM_GPUS} GPU(s), exiting."
#    exit 1
#fi
#echo "Reserved ${MISINFO_NUM_GPUS} GPUs: ${MISINFO_GPUS}"
#MISINFO_TRAIN_GPUS=${MISINFO_GPUS}
#MISINFO_EVAL_GPUS=${MISINFO_GPUS}

DATASET_PATH=data/${DATASET}
#
## trap ctrl+c to free GPUs
#handler()
#{
#    echo "Experiment aborted."
#    echo "Freeing ${MISINFO_NUM_GPUS} GPUs: ${MISINFO_GPUS}"
#    python gpu/free_gpus.py -i ${MISINFO_GPUS}
#    exit 1
#}
#trap handler SIGINT


SPLIT_FILES=""
for (( SPLIT=1; SPLIT<=${NUM_SPLITS}; SPLIT++ )) do
  echo "Training split ${SPLIT} model..."
  MODEL_NAME=MISINFO-${DATASET}-${RUN_NAME}_${RUN_ID}-s${SPLIT}

  if [[ ${MISINFO_TRAIN} = true ]]; then
      echo "Training misinfo model..."
      python rel/train.py \
        --emb_size ${MISINFO_EMB_SIZE} \
        --emb_model ${MISINFO_EMB_MODEL} \
        --emb_loss_norm ${MISINFO_EMB_LOSS_NORM} \
        --train_misinfo_path ${DATASET_PATH}/misinfo.json \
        --val_misinfo_path ${DATASET_PATH}/misinfo.json \
        --train_path ${DATASET_PATH}/train_s"${SPLIT}".jsonl \
        --val_path ${DATASET_PATH}/dev_s"${SPLIT}".jsonl \
        --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
        --model_name "${MODEL_NAME}" \
        --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
        --batch_size ${MISINFO_BATCH_SIZE} \
        --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
        --learning_rate ${MISINFO_LEARNING_RATE} \
        --gamma ${MISINFO_GAMMA} \
        --epochs ${MISINFO_TRAIN_EPOCHS} \
        --gpus ${MISINFO_TRAIN_GPUS}
  fi

  if [[ ${MISINFO_RUN} = true ]]; then
      echo "Running dev misinfo..."
      python rel/evaluate.py \
        --emb_size ${MISINFO_EMB_SIZE} \
        --emb_model ${MISINFO_EMB_MODEL} \
        --emb_loss_norm ${MISINFO_EMB_LOSS_NORM} \
        --eval_mode ${MISINFO_EVAL_MODE} \
        --misinfo_path ${DATASET_PATH}/misinfo.json \
        --val_path ${DATASET_PATH}/dev_s"${SPLIT}".jsonl \
        --test_path ${DATASET_PATH}/test_s"${SPLIT}".jsonl \
        --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
        --model_name "${MODEL_NAME}" \
        --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
        --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
        --gpus ${MISINFO_EVAL_GPUS}
    fi
    SPLIT_FILES="${SPLIT_FILES},models/${MODEL_NAME}/results.json"
done

#echo "Freeing ${MISINFO_NUM_GPUS} GPUs: ${MISINFO_GPUS}"
#python gpu/free_gpus.py -i ${MISINFO_GPUS}

python identify/multi_split_eval_glp.py \
  --input_path ${SPLIT_FILES}
