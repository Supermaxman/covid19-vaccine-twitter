
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


def parse_article(article_dict):
	url = article_dict['url']
	html = article_dict['html']
	try:
		article = Article(url, config=config)
		article.download(html)
		article.parse()
		parsed_article = {
			'url': url,
			'title': article.title,
			'text': article.text
		}
	except Exception as e:
		print(e)
		parsed_article = {
			'url': url,
			'title': '',
			'text': ''
		}

	return parsed_article


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
	parser.add_argument('-s', '--seed', default=0, type=int)
	args = parser.parse_args()

	np.random.seed(args.seed)
	random.seed(args.seed)

	print(f'reading {args.input_path}')
	articles = {}
	article_lines = read_jsonl(args.input_path)
	for article in article_lines:
		url = article['url']
		articles[url] = article
	print(f'Total articles read: {len(articles)}')
	parsed_articles = []
	with Pool(processes=8) as p:
		for p_article in tqdm(p.imap_unordered(parse_article, articles.values()), total=len(articles)):
			parsed_articles.append(p_article)
	print(f'{len(parsed_articles)} articles parsed')
	write_jsonl(parsed_articles, args.output_path)
	print('Done!')
