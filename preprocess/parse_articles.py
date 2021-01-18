
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
	html = article_dict['article_html']
	title = ''
	text = ''
	authors = []
	summary = ''
	try:
		article = Article(url, config=config)
		article.download(html)
		article.parse()
		title = article.title
		text = article.text
		authors = article.authors

		article.nlp()
		summary = article.summary
	except Exception as e:
		print(e)
	try:
		parsed_article = {
			'url': url,
			'title': title,
			'text': text,
			'authors': authors,
			'summary': summary
		}
		parsed_article = json.dumps(parsed_article)
	except:
		parsed_article = None
	return parsed_article


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
	articles = read_jsonl_generator(args.input_path)
	with open(args.output_path, 'w') as f:
		with Pool(processes=8) as p:
			for p_article in tqdm(p.imap_unordered(parse_article, articles), total=31150):
				if p_article is not None:
					f.write(p_article + '\n')
	print('Done!')
