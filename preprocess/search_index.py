
import json
from tqdm import tqdm
import argparse
from collections import defaultdict

from pyserini.search import SimpleSearcher


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--index_path', required=True)
	parser.add_argument('-q', '--query_path', required=True)
	parser.add_argument('-r', '--output_path', required=True)
	parser.add_argument('-k', '--top_k', default=2000, type=int)
	parser.add_argument('-bk1', '--bm25_k1', default=0.82, type=float)
	parser.add_argument('-bb', '--bm25_b', default=0.68, type=float)

	args = parser.parse_args()

	with open(args.query_path) as f:
		misinfo = json.load(f)

	searcher = SimpleSearcher(args.index_path)
	searcher.set_bm25(args.bm25_k1, args.bm25_b)
	print(f'Running search...')

	scores = {}
	for m_id, m in tqdm(misinfo.items()):
		hits = searcher.search(m['text'], k=args.top_k)
		for rank, hit in enumerate(hits[:args.top_k], start=1):
			tweet_id = hit.docid
			if tweet_id not in scores:
				scores[tweet_id] = {}
			scores[hit.docid][m_id] = hit.score

		hits = searcher.search(m['alternate_text'], k=args.top_k)
		for rank, hit in enumerate(hits[:args.top_k], start=1):
			tweet_id = hit.docid
			if tweet_id not in scores:
				scores[tweet_id] = {}
			score = hit.score
			if m_id in scores[hit.docid]:
				# not really proper way to compare bm25 scores, but should be ok for such similar queries for now
				score = max(scores[hit.docid], hit.score)
			scores[hit.docid][m_id] = score

	with open(args.output_path, 'w') as f:
		json.dump(scores, f)

	print('Done!')
