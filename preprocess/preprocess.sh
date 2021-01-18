#!/usr/bin/env bash

#python preprocess/filter_unique.py \
#    --input_path data/raw-v1 \
#    --output_path data/unique-v1.jsonl \
#    --min_jaccard 0.25

#python preprocess/add_references.py \
#    --input_path data/unique-v1.jsonl \
#    --output_path data/unique-ref-v1.jsonl

python preprocess/download_articles.py \
    --input_path data/unique-ref-v1.jsonl \
    --output_path data/unique-ref-articles-v1.jsonl
