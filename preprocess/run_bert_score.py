import os
import json
from collections import defaultdict
from tqdm import tqdm
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
import bert_score
from bert_score import BERTScorer
import torch
import numpy as np


def read_jsonl(path):
	examples = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				try:
					ex = json.loads(line)
					examples.append(ex)
				except Exception as e:
					print(e)
	return examples


def write_jsonl(data, path):
	with open(path, 'w') as f:
		for example in data:
			json_data = json.dumps(example)
			f.write(json_data + '\n')


if __name__ == '__main__':
	input_path = '../data/unique-art-v1.jsonl'
	misinfo_path = '../data/misinfo.json'
	output_path = '../data/scores.json'
	device = 'cuda:4'
	# best results from COVIDLies paper
	model_type = 'digitalepidemiologylab/covid-twitter-bert-v2'
	# decided for bert-large-uncased by BERTScore library experiments
	num_layers = 18
	batch_size = 32
	max_length_percentile = 95

	tweets = read_jsonl(input_path)
	print(f'Total tweets read: {len(tweets)}')
	with open(misinfo_path) as f:
		misinfo = json.load(f)

	# https://arxiv.org/abs/1904.09675
	# https://github.com/Tiiiger/bert_score
	scorer = BERTScorer(
		model_type=model_type,
		num_layers=18,
		device=device
	)
	max_chars = int(np.percentile([len(t['full_text']) for t in tweets], max_length_percentile))
	tweet_texts = []
	m_texts = []
	for t in tweets:
		tweet_id = t['id']
		tweet_text = t['full_text'][:max_chars]
		for m_id, m in misinfo.items():
			m_text = m['text']
			tweet_texts.append(tweet_text)
			m_texts.append(m_text)

	t_p, t_r, t_f1 = scorer.score(
		cands=tweet_texts,
		refs=m_texts,
		verbose=True,
		batch_size=batch_size
	)
	t_f1 = t_f1.view(len(tweets), len(misinfo)).detach().numpy()
	scores = {}
	for t, tweet_scores in zip(tweets, t_f1):
		tweet_id = t['id']
		t_scores = {}
		for (m_id, m), m_score in zip(misinfo.items(), tweet_scores):
			m_score = float(m_score)
			print(f'{tweet_id}: {m_id} - {m_score:.2f}')
			t_scores[m_id] = m_score
		scores[tweet_id] = t_scores

	with open(output_path, 'w') as f:
		json.dump(scores, f)


