
import argparse
import random
import os
import json
import glob
import string
from tqdm import tqdm
import re
import logging
import sys
from multiprocessing import Pool
from collections import defaultdict
from copy import copy

import heapq
import numpy as np
import mmh3


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


def thread_multi_hash_packed(args):
	return thread_multi_hash(*args)


def thread_multi_hash(document, hash_seeds, hash_bits):
	""" Generates a texts minhash signature using multi-hash method.
	Uses i random hashes for j permutations selecting the minimum hash value
	each time to build each texts hash signature.
	Slower but more stable than k smallest hash method.
	Args:
		document (list): List of document shingles.
		hash_seeds (list):
		hash_bits (list):
	Returns:
		list: List of text signatures generated using k smallest neighbours method.
	"""
	signature = []
	for seed in np.nditer(hash_seeds):
		_min_value = None
		for shingle in document:
			if hash_bits == 64:
				hash_value = mmh3.hash64(
					shingle, int(seed)
				)[0]
			elif hash_bits == 32:
				hash_value = mmh3.hash(
					shingle, int(seed)
				)
			else:
				hash_value = mmh3.hash128(
					shingle, int(seed)
				)
			if not _min_value:
				_min_value = hash_value
			elif _min_value > hash_value:
				_min_value = hash_value
		signature.append(_min_value)
	return signature


class MinHash:
	""" MinHash.
	Attributes:
		n_gram (int): Number of characters used in each shingle.
		n_gram_type (str): Type of n gram used for shingles.
		permutations (int): Number of random permutations used to generate signatures.
		hash_bits (int): Hash value size used to generate signatures.
		method (str): Method used to generate signatures.
		seed (int): Seed used to generate signatures.
		signatures (np.array): Matrix of minhash signatures, m represents each texts
			minhash signature with n representing each permutations minimum hash value.
	"""

	def __init__(
			self,
			text,
			n_gram=9,
			n_gram_type='char',
			permutations=100,
			hash_bits=64,
			method='multi_hash',
			seed=None,
			n_jobs=1
	):
		""" Generates a minhash signature matrix for texts in a corpus.
		Args:
			text (list, np.array): Iterable containing text content of each document.
			n_gram (int): Number of characters to be used in each shingle.
			n_gram_type (str): Type of n gram to use for shingles, must be char or term.
			permutations (int): Number of hash values in each document signature.
			hash_bits (int): Hash value size, must be 32, 64 or 128 bit.
			method (str): Method to be used for minhash function, must be multi_hash
				or k_smallest_values.
			seed (int): Seeds from which to generate random hash function.
		"""
		logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
							datefmt='%m/%d/%Y %H:%M:%S',
							level=logging.INFO)
		logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
		self.logger = logging.getLogger(__name__)
		self.n_jobs = n_jobs
		self.n_gram = n_gram
		if n_gram_type not in ['char', 'term']:
			raise ValueError(
				'Only "char" and "term" n_gram types are supported.'
			)
		self.n_gram_type = n_gram_type
		self.permutations = permutations
		if hash_bits not in [32, 64, 128]:
			raise ValueError(
				'Only 32, 64 and 128 bit hashes are supported.'
			)
		self.hash_bits = hash_bits
		if method not in [
			'multi_hash',
			'k_smallest_values'
		]:
			raise ValueError(
				'Only "multi_hash" and "k_smallest_value" hash methods are supported.'
			)
		self.method = method
		self.seed = None
		if seed:
			self.seed = seed
			np.random.seed(seed)
		if method == 'multi_hash':
			self._hash_seeds = np.random.randint(
				low=1, high=100_000_000, size=permutations
			)
		else:
			self._hash_seeds = np.random.randint(
				low=1, high=100_000_000
			)

		self.size = len(text)
		# Run methods.
		self._shingles = self._k_shingles(text, method is 'multi_hash')
		self.signatures = self._min_hash()

	def _k_shingles(self, texts, packed=False):
		""" Generates shingles for each input text.
		Breaks strings into k overlapping shingles consisting of characters or terms
		of n_gram size.
		Args:
			texts (list, np.array): list, array or Pandas series of input texts.
		Yields:
			List: Shingle list generated for each input text.
		"""
		trim_overflow = (self.n_gram - 1) * -1
		if type(texts) == str:
			texts = [texts]
		for text in texts:
			if self.n_gram_type == 'char':
				shingles = [
							   text[char:char + self.n_gram]
							   for char in range(len(text))
						   ][:trim_overflow]
			else:
				terms = text.split()
				shingles = [
							   ' '.join(terms[term:term + self.n_gram])
							   for term in range(len(terms))
						   ][:trim_overflow]
			if not shingles:
				raise ValueError(
					'Shingle "n_gram" size must not exceed minimum text length.'
				)
			if packed:
				yield shingles, self._hash_seeds, self.hash_bits
			else:
				yield shingles

	def _k_smallest_hash(self, document):
		""" Generates a texts minhash signature using k smallest neighbours method.
		Uses a single random hash to simulate a shuffle of each texts shingles.
		Then selecting i smallest minimum hash values for j permutations.
		Faster but less stable than multi hash method.
		Args:
			document (list): List of text shingles.
		Returns:
			list: List of text signatures generated using k smallest neighbours method.
		"""
		signature = []
		# Uses a heap to make calculating n smallest values more efficient.
		heapq.heapify(signature)
		if len(document) <= self.permutations:
			raise ValueError(
				'N permutations must not be >= n shingles for k_smallest_values method'
			)
		for shingle in document:
			if self.hash_bits == 64:
				hashed_shingle = mmh3.hash64(
					shingle, self._hash_seeds
				)[0]
			elif self.hash_bits == 32:
				hashed_shingle = mmh3.hash(
					shingle, self._hash_seeds
				)
			else:
				hashed_shingle = mmh3.hash128(
					shingle, self._hash_seeds
				)
			heapq.heappush(signature, hashed_shingle)
		return heapq.nsmallest(self.permutations, signature)

	def _min_hash(self):
		""" Calculates document signature by calling the selected hashing method.
		Returns:
			 np.array: Matrix of minhash signatures, m represents each texts minhash
				signature with n representing each permutations minimum hash value.
		"""
		if self.method is 'multi_hash':
			signatures = []
			with Pool(self.n_jobs) as p:
				for sig in tqdm(p.imap(thread_multi_hash_packed, self._shingles), total=self.size):
					signatures.append(sig)
			return np.array(signatures)
		else:
			signatures = []
			for document in self._shingles:
				signature = self._k_smallest_hash(document)
				signatures.append(signature)
			return np.array(signatures)


class LSH:
	""" Locality Sensitive Hashing.
	Attributes:
		no_of_bands (int): Number of bands used in model.
		permutations (int): Number of permutations used in MinHash.
	"""

	def __init__(self, minhash=None, labels=None, no_of_bands=None):
		""" Initialize the LSH object.
		Args:
			minhash (np.array): Object returned by MinHash class.
			labels (list, np.array): Iterable, array or pandas series containing labels.
			no_of_bands (int): Number of bands to break minhash signature into.
		"""
		# Create default variables
		self.no_of_bands = no_of_bands
		self._buckets = defaultdict(list)
		self._i_bucket = defaultdict(list)
		self.permutations = None
		# Run methods if minhash and labels provided
		if minhash and labels:
			self.permutations = minhash.permutations
			self._lsh(minhash.signatures, labels)
		elif minhash:
			raise ValueError(
				'labels cannot be None if LSH initialised with minhash object.'
			)
		elif labels:
			raise ValueError(
				'minhash object cannot be None if LSH initialised with labels.'
			)

	def _lsh(self, signatures, labels):
		""" Break signatures into bands and hash components to buckets.
		Args:
			signatures (np.array): MinHash signature Matrix.
			labels (list): List of labels for MinHash signatures.
		"""
		if not self.no_of_bands:
			self.no_of_bands = self.permutations // 2

		for label, signature in tqdm(zip(labels, signatures), total=len(labels)):
			bands = np.hsplit(
				signature,
				self.no_of_bands
			)
			for band in bands:
				bucket_id = hash(tuple(band))
				self._buckets[bucket_id].append(label)
				self._i_bucket[label].append(bucket_id)

	def _candidate_duplicates(self, bucket_ids, label, sensitivity, jaccard):
		""" Identify candidate duplicates and check Jaccard Similarity.
		Args:
			bucket_ids (list): List of bucket ids.
			label (str, int, float): Text label.
			sensitivity (int): Number of identical buckets two ids must occur
				in to be considered a near duplicate pair.
			jaccard (float): Minimum Jaccard Similarity for documents to be
				counted as near duplicates.
		Returns:
			List: Near duplicate document ids.
		"""
		candidates = defaultdict(int)
		# Retrieve candidate duplicate pairs from model.
		for bucket_id in bucket_ids:
			matches = copy(self._buckets.get(bucket_id))
			matches.remove(label)
			for match in matches:
				candidates[match] += 1
		# Apply sensitivity threshold.
		if sensitivity > 1:
			for key in list(candidates):
				if candidates[key] < sensitivity:
					del candidates[key]
		# Apply Jaccard threshold and unzip pairs.
		if jaccard:
			for key in list(candidates):
				jaccard_ratio = candidates[key] / self.no_of_bands
				if jaccard_ratio < jaccard:
					del candidates[key]
		candidates = list(candidates)
		return candidates

	def update(self, minhash, new_labels):
		""" Updates LSH object with new MinHash matrix and labels.
		Args:
			minhash (minhash): MinHash object containing new minhash signatures to
				add to LSH object.
			new_labels (list): List of new labels to add to LSH object.
		"""
		if self._i_bucket:
			# Check if texts already exist in model.
			if set(
					self._i_bucket.keys()
			).intersection(
				set(new_labels)
			) != set():
				raise ValueError(
					'At least one provided label already exists in model.'
				)
			if self.permutations != minhash.permutations:
				raise ValueError(
					'Number of permutations in minhash must be {} to match LSH model.'.format(
						self.permutations
					)
				)
		else:
			# Create parameters for new model.
			self.permutations = minhash.permutations
		# Update model.
		self._lsh(minhash.signatures, new_labels)

	def query(self, label, min_jaccard=None, sensitivity=1):
		""" Returns near duplicates from model.
		Takes a provided text label and returns a list of labels for texts whose
		similarity with the provided text is above a provided threshold.
		Can be used to create a recommendation model.
		Args:
			label (str, int, float): Label of text for which to return near duplicates.
			min_jaccard (float): Minimum Jaccard Similarity for texts to be returned as
				near duplicates.
			sensitivity (int): Number of unique buckets two ids must co-occur in to be
				considered a near duplicate pair.
		Returns:
			List: Candidate duplicates for provided text label.
		"""
		if sensitivity > self.no_of_bands:
			raise ValueError(
				'Sensitivity must be <= no of bands.'
			)
		buckets = self._i_bucket.get(label)
		if not buckets:
			raise KeyError(
				'Label {} does not exist in model'.format(label)
			)
		return self._candidate_duplicates(
			buckets, label, sensitivity, min_jaccard
		)

	def remove(self, label):
		""" Remove label and associated text signature from model.
		Args:
			label (str, int, float): Label for text to be removed from model.
		"""
		buckets = self._i_bucket.get(label)
		if not buckets:
			raise KeyError(
				'Label {} does not exist in model.'.format(label)
			)
		for bucket in buckets:
			self._buckets[bucket].remove(label)
			if not self._buckets[bucket]:
				del self._buckets[bucket]
		del self._i_bucket[label]

	def contains(self):
		""" Returns a list of all labels contained in the model.
		Returns:
			 List: All labels for texts contained in the model.
		"""
		return list(self._i_bucket)

	def adjacency_list(self, min_jaccard=None, sensitivity=1):
		""" Returns adjacency list.
		Iterates over texts, pairing each text with a list of labels whose relationships with
		each text are above a certain threshold.
		Can be used to create an undirected graph for texts in the LSH object.
		Args:
			min_jaccard (float): Minimum Jaccard Similarity for texts to be returned as near
				duplicates.
			sensitivity (int): Number of unique buckets two ids must co-occur in to be
				considered a near duplicate pair.
		Returns:
			Dict: Adjacency list.
		"""
		if sensitivity > self.no_of_bands:
			raise ValueError(
				'Sensitivity must be <= no of bands.'
			)
		adjacency_list = {}
		for label in self._i_bucket.keys():
			buckets = self._i_bucket.get(label)
			candidates = self._candidate_duplicates(
				buckets, label, sensitivity, min_jaccard
			)
			adjacency_list[label] = candidates
		return adjacency_list

	def edge_list(
			self,
			min_jaccard=0,
			jaccard_weighted=False,
			sensitivity=1
	):
		""" Returns list of relationship pairs between related texts.
		Iterates over texts to create relationship pairs from hash bucket contents, where
		relationships are above a certain threshold.
		Edge list can be used to create an undirected graph, optionally with edges weighted
		by Jaccard similarity.
		May be slow and scale poorly for larger corpora.
		Args:
			min_jaccard (float): Minimum Jaccard Similarity for relationship to be returned.
			jaccard_weighted (bool): If True return a list of 3 tuples including the
				relationship pairs and their associated Jaccard similarity.
			sensitivity (int): Number of unique buckets two ids must co-occur for relationship
				to be returned.
		Returns:
			List: 2 tuple relationship pairs between texts, optionally a weighted 3 tuple.
		"""
		if sensitivity > self.no_of_bands:
			raise ValueError(
				'Sensitivity must be <= no of bands.'
			)
		edges = []
		labels = list(self._i_bucket)
		for i in range(len(labels)):
			candidates = defaultdict(int)
			label = labels.pop()
			for bucket in self._i_bucket.get(label):
				matches = copy(self._buckets.get(bucket))
				matches.remove(label)
				for match in matches:
					candidates[match] += 1
			if sensitivity > 1:
				for key in list(candidates):
					if candidates[key] < sensitivity:
						del candidates[key]
			for candidate in list(candidates):
				if candidate in labels:
					if min_jaccard or jaccard_weighted:
						jaccard_ratio = candidates[candidate] / self.no_of_bands
						if jaccard_ratio >= min_jaccard:
							if jaccard_weighted:
								edges.append(
									(label, candidate, jaccard_ratio)
								)
							else:
								edges.append(
									(label, candidate)
								)
					else:
						edges.append(
							(label, candidate)
						)
		return edges


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-j', '--min_jaccard', default=0.25, type=float)
	parser.add_argument('-s', '--seed', default=0, type=int)
	args = parser.parse_args()

	np.random.seed(args.seed)
	random.seed(args.seed)
	tweets = {}
	for file_name in os.listdir(args.input_path):
		file_path = os.path.join(args.input_path, file_name)
		print(f'reading {file_path}')
		tweet_lines = read_jsonl(file_path)
		file_tweets = 0
		for tweet_line in tweet_lines:
			if 'data' in tweet_line and 'id' in tweet_line['data']:
				tweet_id = tweet_line['data']['id']
				tweets[tweet_id] = tweet_line
				file_tweets += 1
		print(f'{file_tweets} read.')
	print(f'Total tweets read: {len(tweets)}')

	transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
	all_text = []
	t_map = {}
	for t_idx, (tweet_id, tweet) in enumerate(tqdm(list(tweets.items()))):
		tweet_text = tweet['data']['text'].translate(transl_table).lower().translate(
			str.maketrans('', '', string.punctuation)).strip().replace(',', '')
		tweet_text = tweet_text.replace('\n', ' ').replace('\t', ' ')
		all_text.append(tweet_text)
		t_map[t_idx] = tweet_id

	print('Min hashing...')
	minhash = MinHash(
		all_text,
		n_gram=9,
		n_gram_type='char',
		permutations=100,
		hash_bits=64,
		seed=args.seed,
		n_jobs=8
	)

	print('Constructing LSH...')
	lsh = LSH(
			minhash,
			list(range(len(tweets)))
	)

	print('Finding duplicates...')
	seen_idxs = set()
	unique_tweets = {}
	for t_idx, (tweet_id, tweet) in enumerate(tqdm(tweets.items())):
		closest_tweets = lsh.query(
			t_idx,
			min_jaccard=args.min_jaccard
		)
		duplicate = False
		if len(closest_tweets) > 0:
			close_id = None
			duplicate_ids = []
			for close_idx in closest_tweets:
				if close_idx in seen_idxs:
					duplicate = True
					close_id = t_map[close_idx]
					duplicate_ids.append(close_id)
			tweet['duplicates'] = duplicate_ids
			tweet['is_duplicate'] = duplicate
			# if duplicate:
			# 	print('----')
			# 	print(tweet['data']['text'])
			# 	print()
			# 	close_tweet = tweets[close_id]
			# 	print(close_tweet['data']['text'])
			# 	continue
		if not duplicate:
			seen_idxs.add(t_idx)
			unique_tweets[tweet_id] = tweet
	print(f'Total unique tweets: {len(unique_tweets)}')

	print('Writing tweets...')
	write_jsonl(
		unique_tweets.values(),
		args.output_path
	)

	print('Done!')
