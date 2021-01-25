
import argparse
import random
import os
import json
import string
from tqdm import tqdm
import re
from multiprocessing import Pool

import numpy as np
from newspaper import Article, Config


config = Config()
config.fetch_images = False


def download_article(url):
	try:
		article = Article(url, config=config)
		article.download()
		article_html = article.html
		article_text = json.dumps({'url': url, 'article_html': article_html})
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


def read_jsonl_generator(path):
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				try:
					ex = json.loads(line)
					yield ex
				except Exception as e:
					print(e)


def write_jsonl(data, path):
	with open(path, 'w') as f:
		for example in data:
			json_data = json.dumps(example)
			f.write(json_data + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-s', '--seed', default=0, type=int)
	args = parser.parse_args()

	np.random.seed(args.seed)
	random.seed(args.seed)

	print(f'reading {args.input_path}')
	external_urls = set()
	tweets_list = read_jsonl_generator(args.input_path)
	for tweet in tweets_list:
		for t_url, t_url_info in tweet['urls'].items():
			if t_url_info['type'] == 'external' and not t_url_info['quoted']:
				external_urls.add(t_url)
	print(f'{len(external_urls)} external URLs')
	if os.path.exists(args.output_path):
		read_urls = 0
		article_lines = read_jsonl_generator(args.output_path)
		for article_line in article_lines:
			url = article_line['url']
			if url in external_urls:
				external_urls.remove(url)
				read_urls += 1
		print(f'{read_urls} articles already downloaded.')
	external_urls = sorted(list(external_urls))
	num_downloaded = 0
	with open(args.output_path, 'a') as f:
		with Pool(processes=8) as p:
			for url, article_text in tqdm(p.imap_unordered(download_article, external_urls), total=len(external_urls)):
				if article_text is not None:
					num_downloaded += 1
					f.write(article_text + '\n')

	print(f'{num_downloaded} articles downloaded')
	print('Done!')
