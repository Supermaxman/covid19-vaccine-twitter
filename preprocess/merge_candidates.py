
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
	parser.add_argument('-a', '--alternate_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	tweets = read_jsonl(args.input_path)
	tweets = {t['id']: t for t in tweets}
	print(f'Total tweets read: {len(tweets)}')

	alt_tweets = read_jsonl(args.alternate_path)
	alt_tweets = {t['id']: t for t in alt_tweets}
	print(f'Total alt tweets read: {len(alt_tweets)}')
	tweet_ids = set(list(tweets.keys()) + list(alt_tweets.keys()))
	merged_tweets = []
	pair_count = 0
	for tweet_id in tweet_ids:
		if tweet_id in tweets and tweet_id in alt_tweets:
			tweet = tweets[tweet_id]
			alt_tweet = alt_tweets[tweet_id]
			tweet_candidates = tweet['candidates']
			alt_candidates = alt_tweet['candidates']
			merged_candidates = {}
			for m_id in set(list(tweet_candidates.keys()) + list(alt_candidates.keys())):
				if m_id in tweet_candidates and m_id in alt_candidates:
					max_candidate = max(
						[
							tweet_candidates[m_id],
							alt_candidates[m_id]
						],
						key=lambda x: x['score']
					)
					merged_candidates[m_id] = max_candidate
				elif m_id in tweet_candidates:
					merged_candidates[m_id] = tweet_candidates[m_id]
				else:
					merged_candidates[m_id] = alt_candidates[m_id]
			tweet['candidates'] = merged_candidates
		elif tweet_id in tweets:
			tweet = tweets[tweet_id]
		else:
			tweet = alt_tweets[tweet_id]
		merged_tweets.append(tweet)
		pair_count += len(tweet['candidates'])

	print(f'Total merged tweets: {len(merged_tweets)}')
	print(f'Total merged pairs: {pair_count}')
	write_jsonl(merged_tweets, args.output_path)

