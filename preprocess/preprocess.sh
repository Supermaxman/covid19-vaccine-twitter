#!/usr/bin/env bash

python preprocess/filter_unique.py \
    --input_path data/raw-v1 \
    --output_path data/unique-v1.jsonl \
    --min_jaccard 0.25
