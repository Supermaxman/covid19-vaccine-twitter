#!/usr/bin/env bash

python filter_unique.py \
    --input_path /users/max/code/covid19-vaccine-twitter/data/raw-v1 \
    --output_path /users/max/code/covid19-vaccine-twitter/data/unique-v1.jsonl \
    --min_jaccard 0.25
