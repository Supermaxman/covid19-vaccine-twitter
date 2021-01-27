
import argparse
import random
import os
import json
import string
from tqdm import tqdm
import re
from multiprocessing import Pool, Manager
import requests
import time

import numpy as np
# from newspaper import Article, Config


# manager = Manager()
# articles = manager.dict()
# config = Config()
# config.fetch_images = False
transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
url_pattern = re.compile(r'(https:\/\/t\.co\/[\w]*\b)( QT)?')
prefix = 'https://t.co/'
len_url = len(prefix) + 10


def parse_tweet(t):
	tweet_id, tweet = t
	tweet['id'] = tweet_id
	tweet['id_str'] = tweet_id
	if 'tweets' in tweet['includes']:
		ref_tweet_map = {r_t['id']: r_t for r_t in tweet['includes']['tweets']}
	else:
		ref_tweet_map = {}
	ref_user_map = {u['id']: u for u in tweet['includes']['users']}
	tweet_text = tweet['data']['text'].translate(transl_table)
	if 'referenced_tweets' in tweet['data']:
		referenced_tweets = tweet['data']['referenced_tweets']
	else:
		referenced_tweets = []
	if len(referenced_tweets) > 0:
		for r_tweet in referenced_tweets:
			if r_tweet['type'] == 'quoted':
				r_t = ref_tweet_map[r_tweet['id']]
				r_t_text = r_t['text'].translate(transl_table)
				tweet_text = f'{tweet_text} QT: \"{r_t_text}\"'
			elif r_tweet['type'] == 'replied_to':
				pass
	urls = {}
	contains_quote = '\"' in tweet_text
	for url, qt in re.findall(url_pattern, tweet_text):
		if qt != '':
			url_type = 'quote'
		else:
			url_type = 'external'
			# resp = requests.head(url)
			# code = resp.status_code
			# # redirect
			# if code == 301:
			# 	url = resp.headers['Location']
			# 	url_type = 'external'
			# else:
			# 	url_type = 'unknown'
			# time.sleep(0.1)
		urls[url] = {
			'url': url,
			'type': url_type,
			'quoted': contains_quote
		}

	tweet['full_text'] = tweet_text
	tweet['urls'] = urls
	return tweet_id, tweet


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
	parser.add_argument('-o', '--output_path', required=True)
	# parser.add_argument('-a', '--article_cache', required=True)
	parser.add_argument('-s', '--seed', default=0, type=int)
	args = parser.parse_args()

	np.random.seed(args.seed)
	random.seed(args.seed)

	print(f'reading {args.input_path}')
	tweets = {}
	tweet_list = read_jsonl(args.input_path)
	for tweet in tweet_list:
		tweet_id = tweet['data']['id']
		tweets[tweet_id] = tweet
	print(f'Total tweets read: {len(tweets)}')

	print('Adding tweet references...')
	with open(args.output_path, 'w') as f:
		with Pool(processes=8) as p:
			for tweet_id, tweet in tqdm(p.imap_unordered(parse_tweet, tweets.items()), total=len(tweets)):
				f.write(json.dumps(tweet) + '\n')

	print('Done!')
