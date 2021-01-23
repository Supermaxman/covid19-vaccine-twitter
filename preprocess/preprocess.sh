#!/usr/bin/env bash

#python preprocess/filter_unique.py \
#    --input_path data/raw-v1 \
#    --output_path data/unique-v1.jsonl \
#    --min_jaccard 0.25

#python preprocess/add_references.py \
#    --input_path data/unique-v1.jsonl \
#    --output_path data/unique-ref-v1.jsonl

#python preprocess/download_articles.py \
#    --input_path data/unique-ref-v1.jsonl \
#    --output_path data/unique-ref-articles-v1.jsonl
#
#python preprocess/parse_articles.py \
#    --input_path data/unique-ref-articles-v1.jsonl \
#    --output_path data/unique-ref-parsed-articles-v1.jsonl
#
#python preprocess/add_articles.py \
#    --input_path data/unique-ref-v1.jsonl \
#    --articles_path data/unique-ref-parsed-articles-v1.jsonl \
#    --output_path data/unique-art-v1.jsonl

python preprocess/run_bert_score.py \
    --input_path data/unique-art-v1.jsonl \
    --misinfo_path data/misinfo.json \
    --output_path data/scores.json \
    --device cuda:4

