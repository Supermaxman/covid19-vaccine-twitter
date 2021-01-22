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
	model_type = 'digitalepidemiologylab/covid-twitter-bert-v2'
	device = 'cuda:4'
	tweets = read_jsonl(input_path)
	print(f'Total tweets read: {len(tweets)}')
	with open('../data/misinfo.json') as f:
		misinfo = json.load(f)

	scorer = BERTScorer(
		model_type=model_type,
		num_layers=18,
		device=device
	)
	max_chars = int(np.percentile([len(t['full_text']) for t in tweets], 95))
	tweet_ids = [t['id'] for t in tweets]
	m_ids = [m_id for m_id in misinfo]
	t_p, t_r, t_f1 = scorer.score(
		cands=[t['full_text'][:max_chars] for t in tweets],
		refs=[m['text'] for m_id, m in misinfo.items()],
		verbose=True,
		batch_size=8
	)

