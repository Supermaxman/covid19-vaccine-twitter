
import argparse
import random
import os
import json
import string
from tqdm import tqdm
import re
from multiprocessing import Pool, Manager

import numpy as np
# from newspaper import Article, Config


# manager = Manager()
# articles = manager.dict()
# config = Config()
# config.fetch_images = False
transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
url_pattern = re.compile(r'(https:\/\/t\.co\/[\w]*\b) (QT)?')
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
		urls[url] = {
			'url': url,
			'type': 'qt' if qt != '' else 'external',
			'quoted': contains_quote
		}

	# while True:
	# 	idx = tweet_text.find(prefix)
	# 	if idx == -1:
	# 		break
	#
	# 	if tweet_text[idx+len_url+1:idx+len_url+3] == 'QT':
	# 		tweet_text = tweet_text[:idx] + ' ' + tweet_text[idx+len_url:]
	# 	else:
	# 		url = tweet_text[idx:idx+len_url]
	# left_text = tweet_text[:idx]
	# right_text = tweet_text[idx+len_url:]
	# left_quote = any(c == '\"' or c == '\'' for c in left_text)
	# right_quote = any(c == '\"' or c == '\'' for c in right_text)
	# left_s_quote = any(c == '\"' for c in left_text)
	# right_s_quote = any(c == '\"' for c in right_text)
	# a_text = ''
	# if not (left_quote and right_quote) or not (left_s_quote or right_s_quote):
	# 	try:
	# 		if url in articles:
	# 			article = articles[url]
	# 		else:
	# 			article = Article(url, config=config)
	# 			article.download()
	# 			article.parse()
	# 			articles[url] = article
	# 		a_text = article.title.translate(transl_table)
	# 		a_check = a_text.lower().translate(str.maketrans('', '', string.punctuation))
	# 		t_check = tweet_text.lower().translate(str.maketrans('', '', string.punctuation))
	# 		if a_check not in t_check:
	# 			a_text = f'\"{a_text}\"'
	# 	except:
	# 		pass

	# urls.append(url)
	# tweet_text = tweet_text[:idx] + 'URL' + ' ' + a_text + ' ' + tweet_text[idx+len_url:]

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

	# if os.path.exists(args.article_cache):
	# 	print('Adding tweet cached articles...')
	# 	with open(args.article_cache, 'r') as f:
	# 		article_cache = json.load(f)
	# 		for url, article_html in article_cache.items():
	# 			article = Article(url, config=config)
	# 			article.download(article_html)
	# 			article.parse()
	# 			articles[url] = article

	print('Adding tweet references...')
	adjusted_tweets = {}
	with Pool(processes=8) as p:
		for tweet_id, tweet in tqdm(p.imap(parse_tweet, tweets.items()), total=len(tweets)):
			adjusted_tweets[tweet_id] = tweet

	print('Writing tweets...')
	write_jsonl(
		adjusted_tweets.values(),
		args.output_path
	)
	# article_cache = {}
	# print('Saving articles...')
	# for url, article in articles.items():
	# 	article_cache[url] = article.html

	# with open(args.article_cache, 'w') as f:
	# 	json.dump(article_cache, f)

	print('Done!')
