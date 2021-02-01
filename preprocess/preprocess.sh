#!/usr/bin/env bash

python preprocess/filter_unique.py \
    --input_path data/raw-v1 \
    --output_path data/unique-v1.jsonl \
    --min_jaccard 0.5

python preprocess/add_references.py \
    --input_path data/unique-v1.jsonl \
    --output_path data/unique-ref-v1.jsonl

#python preprocess/download_articles.py \
#    --input_path data/unique-ref-v1.jsonl \
#    --output_path data/unique-ref-articles-v1.jsonl
#
#python preprocess/parse_articles.py \
#    --input_path data/unique-ref-articles-v1.jsonl \
#    --output_path data/unique-ref-parsed-articles-v1.jsonl

python preprocess/add_articles.py \
    --input_path data/unique-ref-v1.jsonl \
    --articles_path data/unique-ref-parsed-articles-v1.jsonl \
    --output_path data/unique-art-v1.jsonl


python preprocess/convert_tweets_to_jsonl.py \
    --input_path data/unique-art-v1.jsonl \
    --output_path data/unique-art-v1-index.jsonl

python -m pyserini.index \
    -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator \
    -threads 12 \
    -input data/unique-art-v1-index.jsonl \
    -index data/unique-v1 \
    -storePositions \
    -storeDocvectors \
    -storeRaw

python preprocess/search_index.py \
    --index_path data/unique-v1 \
    --query_path data/misinfo.json \
    --output_path data/bm25-scores-v1.json \
    --top_k 200

python preprocess/select_candidates.py \
    --input_path data/unique-art-v1.jsonl \
    --misinfo_path data/misinfo.json \
    --score_path data/bm25-scores-v1.json \
    --output_path data/unique-art-v1-bm25-candidates.jsonl \
    --top_k 200

python preprocess/run_bert_score.py \
    --input_path data/unique-art-v1.jsonl \
    --misinfo_path data/misinfo.json \
    --output_path data/scores.json \
    --device cuda:4 \
    --batch_size 32

python preprocess/select_candidates.py \
    --input_path data/unique-art-v1.jsonl \
    --misinfo_path data/misinfo.json \
    --score_path data/scores.json \
    --output_path data/unique-art-v1-candidates.jsonl \
    --top_k 200

python preprocess/run_bert_score.py \
    --input_path data/unique-art-v1.jsonl \
    --misinfo_path data/misinfo.json \
    --misinfo_text_type alternate_text \
    --output_path data/alternate-scores.json \
    --device cuda:4 \
    --batch_size 32

python preprocess/select_candidates.py \
    --input_path data/unique-art-v1.jsonl \
    --misinfo_path data/misinfo.json \
    --misinfo_text_type alternate_text \
    --score_path data/alternate-scores.json \
    --output_path data/unique-art-v1-candidates-alternate.jsonl \
    --top_k 200

python preprocess/merge_candidates.py \
    --input_path data/unique-art-v1-candidates.jsonl \
    --alternate_path data/unique-art-v1-candidates-alternate.jsonl \
    --output_path data/unique-art-v1-candidates-merged.jsonl

