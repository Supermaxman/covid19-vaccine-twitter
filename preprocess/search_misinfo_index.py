
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
		tweets = json.load(f)

	searcher = SimpleSearcher(args.index_path)
	searcher.set_bm25(args.bm25_k1, args.bm25_b)
	print(f'Running search...')

	scores = {}
	for t in tqdm(tweets):
		t_id = t['id']
		t_text = t['full_text']
		scores[t_id] = {}
		hits = searcher.search(t_text, k=args.top_k)
		for rank, hit in enumerate(hits[:args.top_k], start=1):
			m_id = hit.docid
			scores[t_id][m_id] = hit.score

	with open(args.output_path, 'w') as f:
		json.dump(scores, f)

	print('Done!')
