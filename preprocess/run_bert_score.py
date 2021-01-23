
import json
import argparse
import logging
import transformers
from bert_score import BERTScorer
import numpy as np
import random


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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-m', '--misinfo_path', required=True)
	parser.add_argument('-gpu', '--device', default='cuda:0')
	# best results from COVIDLies paper
	parser.add_argument('-mt', '--model_type', default='digitalepidemiologylab/covid-twitter-bert-v2')
	# decided for bert-large-uncased by BERTScore library experiments
	parser.add_argument('-ml', '--num_layers', default=18, type=int)
	parser.add_argument('-bs', '--batch_size', default=32, type=int)
	parser.add_argument('-mlp', '--max_length_percentile', default=95, type=int)
	parser.add_argument('-s', '--seed', default=0, type=int)
	args = parser.parse_args()

	np.random.seed(args.seed)
	random.seed(args.seed)

	transformers.tokenization_utils.logger.setLevel(logging.ERROR)
	transformers.configuration_utils.logger.setLevel(logging.ERROR)
	transformers.modeling_utils.logger.setLevel(logging.ERROR)

	tweets = read_jsonl(args.input_path)
	print(f'Total tweets read: {len(tweets)}')
	with open(args.misinfo_path) as f:
		misinfo = json.load(f)

	# https://arxiv.org/abs/1904.09675
	# https://github.com/Tiiiger/bert_score
	scorer = BERTScorer(
		model_type=args.model_type,
		num_layers=args.num_layers,
		device=args.device
	)
	max_chars = int(np.percentile([len(t['full_text']) for t in tweets], args.max_length_percentile))
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
		batch_size=args.batch_size
	)
	t_f1 = t_f1.view(len(tweets), len(misinfo)).detach().numpy()
	scores = {}
	for t, tweet_scores in zip(tweets, t_f1):
		tweet_id = t['id']
		t_scores = {}
		for (m_id, m), m_score in zip(misinfo.items(), tweet_scores):
			m_score = float(m_score)
			t_scores[m_id] = m_score
		scores[tweet_id] = t_scores

	with open(args.output_path, 'w') as f:
		json.dump(scores, f)


