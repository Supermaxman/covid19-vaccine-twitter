#!/usr/bin/env bash

filename=$(basename -- "$0")
# run names
RUN_ID=${filename::-3}
RUN_NAME=HLTRI_COVID_MISINFO

# collection
DATASET=v1

# major hyper-parameters for system
MISINFO_PRE_MODEL_NAME=digitalepidemiologylab/covid-twitter-bert-v2
#MISINFO_THRESHOLD=0.2

MISINFO_BATCH_SIZE=8
MISINFO_MODEL_TYPE=lm-avg
MISINFO_TRAIN_SAMPLING=negative
MISINFO_MAX_SEQ_LEN=96
MISINFO_EMB_SIZE=32
MISINFO_LEARNING_RATE=5e-4
MISINFO_TRAIN_EPOCHS=10
MISINFO_EVAL_BATCH_SIZE=4

MISINFO_NUM_GPUS=1
MISINFO_TRAIN=true
MISINFO_RUN=true
MISINFO_EVAL=true

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
MISINFO_SPLIT_FILES=""

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
    python identify/train.py \
      --model_type ${MISINFO_MODEL_TYPE} \
      --emb_size ${MISINFO_EMB_SIZE} \
      --misinfo_path ${DATASET_PATH}/misinfo.json \
      --train_path ${DATASET_PATH}/train.jsonl \
      --val_path ${DATASET_PATH}/dev.jsonl \
      --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
      --model_name MISINFO-${DATASET}-${RUN_NAME}_${RUN_ID} \
      --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
      --batch_size ${MISINFO_BATCH_SIZE} \
      --eval_batch_size ${MISINFO_EVAL_BATCH_SIZE} \
      --train_sampling ${MISINFO_TRAIN_SAMPLING} \
      --learning_rate ${MISINFO_LEARNING_RATE} \
      --epochs ${MISINFO_TRAIN_EPOCHS} \
      --fine_tune \
      --gpus ${MISINFO_TRAIN_GPUS}
fi

#if [[ ${MISINFO_RUN} = true ]]; then
#    echo "Running misinfo..."
#    python identify/predict.py \
#      --model_type lm-gcn-expanded \
#      --create_edge_features \
#      --lex_edge_expanded dep,pos \
#      --graph_names semantic,emotion,lexical \
#      --gcn_size 64 \
#      --gcn_depth 6 \
#      --gcn_type attention \
#      --misconception_info_path ${DATASET_PATH}/misconceptions_extra.json \
#      --split_path ${DATASET_PATH}/${SPLIT_TYPE}_split_${SPLIT}.json \
#      --pre_model_name ${MISINFO_PRE_MODEL_NAME} \
#      --model_name MISINFO-${DATASET}-${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID} \
#      --output_path ${ARTIFACTS_PATH}/${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID} \
#      --max_seq_len ${MISINFO_MAX_SEQ_LEN} \
#      --batch_size ${MISINFO_BATCH_SIZE} \
#      --load_trained_model \
#      --gpus ${MISINFO_EVAL_GPUS} \
#    ; \
#    python identify/format_predictions.py \
#      --input_path ${ARTIFACTS_PATH}/${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID} \
#      --output_path ${ARTIFACTS_PATH}/${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID}/predictions.MISINFO
#fi

#MISINFO_SPLIT_FILES="${MISINFO_SPLIT_FILES},${ARTIFACTS_PATH}/${RUN_NAME}_SPLIT_${SPLIT}_${RUN_ID}/predictions.MISINFO"

echo "Freeing ${MISINFO_NUM_GPUS} GPUs: ${MISINFO_GPUS}"
python gpu/free_gpus.py -i ${MISINFO_GPUS}

#if [[ ${MISINFO_EVAL} = true ]]; then
#    echo "Evaluating misinfo model..."
#    mkdir -p ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}
#    python identify/format_eval.py \
#      --input_path ${MISINFO_SPLIT_FILES} \
#      --output_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/all.run \
#      --threshold ${MISINFO_THRESHOLD}
#
#    python identify/eval.py \
#      --label_path ${DATASET_PATH}/downloaded_tweets_labeled.jsonl \
#      --run_path ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/all.run \
#      > ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/eval.txt \
#      ; \
#      cat ${ARTIFACTS_PATH}/${RUN_NAME}_${RUN_ID}/eval.txt
#fi


