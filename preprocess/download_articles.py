
import argparse
import random
import os
import json
import string
from tqdm import tqdm
import re
from multiprocessing import Pool
import csv

import numpy as np
from newspaper import Article, Config


config = Config()
config.fetch_images = False


def download_article(url):
	try:
		article = Article(url, config=config)
		article.download()
		article_html = article.html
		article_text = article_html
	except:
		article_text = None

	return url, article_text


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

	external_urls = set()
	for tweet_id, tweet in tweets.items():
		for t_url, t_url_info in tweet['urls'].items():
			if t_url_info['type'] == 'external' and not t_url_info['quoted']:
				external_urls.add(t_url)
	print(f'{len(external_urls)} external URLs')
	articles = {}
	if os.path.exists(args.output_path):
		read_urls = 0
		article_lines = read_jsonl(args.output_path)
		for article_line in article_lines:
			url = article_line['url']
			external_urls.remove(url)
			read_urls += 1
		print(f'{read_urls} articles already downloaded.')
	external_urls = sorted(list(external_urls))
	with open(args.output_path, 'a') as f:
		writer = csv.writer(f, delimiter=',', quotechar='|')
		with Pool(processes=8) as p:
			for url, article_text in tqdm(p.imap_unordered(download_article, external_urls), total=len(external_urls)):
				if article_text is not None:
					writer.writerow([url, article_text])

	print(f'{len(articles)} articles downloaded')
	print('Done!')
