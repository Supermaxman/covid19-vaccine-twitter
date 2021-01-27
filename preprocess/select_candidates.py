
import os
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import argparse


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
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-m', '--misinfo_path', required=True)
	parser.add_argument('-sc', '--score_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-k', '--top_k', default=100, type=int)
	args = parser.parse_args()

	tweets = read_jsonl(args.input_path)
	tweets = {t['id']: t for t in tweets}

	print(f'Total tweets read: {len(tweets)}')
	with open(args.misinfo_path) as f:
		misinfo = json.load(f)

	with open(args.score_path) as f:
		scores = json.load(f)

	misinfo_scores = defaultdict(list)
	for tweet_id, t_scores in scores.items():
		for m_id, m_score in t_scores.items():
			misinfo_scores[m_id].append((m_score, tweet_id))

	print(f'Sorting top-{args.top_k} tweets for each misinformation target...')
	for m_id in misinfo_scores:
		misinfo_scores[m_id] = sorted(
			misinfo_scores[m_id],
			# (score, tweet_id)
			key=lambda x: x[0],
			reverse=True
		)
	candidate_ids = set()
	candidate_tweets = []
	for m_id, m in misinfo.items():
		m_rel = misinfo_scores[m_id][:args.top_k]
		rank = 1
		for t_score, tweet_id in m_rel:
			tweet = tweets[tweet_id]
			if 'candidates' not in tweet:
				tweet['candidates'] = {}
			tweet['candidates'][m_id] = {
				'text': m['text'],
				'rank': rank,
				'score': t_score
			}
			rank += 1
			if tweet_id not in candidate_ids:
				candidate_tweets.append(tweet)
				candidate_ids.add(tweet_id)

	print(f'Total candidate tweets: {len(candidate_tweets)}')
	write_jsonl(candidate_tweets, args.output_path)

