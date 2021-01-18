
import argparse
import random
import os
import json
import string
from tqdm import tqdm
import re

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
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-a', '--articles_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
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

	print(f'reading {args.articles_path}')
	articles = {}
	article_list = read_jsonl(args.articles_path)
	for article in article_list:
		url = article['url']
		articles[url] = article

	print(f'adding articles to tweets...')
	transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-\n\t", u"'''\"\"--  ")])
	for tweet_id, tweet in tqdm(tweets.items(), total=len(tweets)):
		tweet_text = tweet['full_text']
		for t_url, t_url_info in tweet['urls'].items():
			t_replace_text = 'URL'
			if t_url in articles:
				t_article = articles[t_url]
				a_text = t_article['title'].translate(transl_table)
				a_check = a_text.lower().translate(str.maketrans('', '', string.punctuation))
				t_check = tweet_text.lower().translate(str.maketrans('', '', string.punctuation))
				if a_check not in t_check:
					t_replace_text += f': \"{a_text}\"'
			tweet_text = tweet_text.replace(t_url, t_replace_text)
		tweet['full_text'] = tweet_text

	print('Writing tweets...')
	write_jsonl(tweets.values(), args.output_path)

	print('Done!')
