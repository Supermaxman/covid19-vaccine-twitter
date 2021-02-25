
import json
import os
import json
from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
import random
from collections import defaultdict
import numpy as np
import string
import spacy
import pickle
import zlib


def read_jsonl(path):
	examples = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				examples.append(ex)
	return examples


def write_jsonl(data, path):
	with open(path, 'w') as f:
		for example in data:
			json_data = json.dumps(example)
			f.write(json_data + '\n')


def load_dataset(split_path, dataset_args, name):
	args_string = str(zlib.adler32(str(dataset_args).encode('utf-8')))

	cache_path = split_path + f'_{name}_{args_string}.cache'
	if os.path.exists(cache_path):
		with open(cache_path, 'rb') as f:
			dataset = pickle.load(f)
	else:
		dataset = MisinfoDataset(
			**dataset_args
		)
		with open(cache_path, 'wb') as f:
			pickle.dump(dataset, f)
	return dataset


def label_text_to_stance_id(label):
	# 'agree', 'disagree', 'no_stance', 'not_relevant',
	if label == 'not_relevant':
		return 0
	if label == 'agree':
		return 1
	if label == 'disagree':
		return 2
	if label == 'no_stance':
		return 3
	else:
		raise ValueError(f'Unknown label: {label}')


def label_text_to_relevant_id(label):
	# 'agree', 'disagree', 'no_stance', 'not_relevant',
	if label == 'not_relevant':
		return 0
	if label == 'agree':
		return 1
	if label == 'disagree':
		return 1
	if label == 'no_stance':
		return 1
	else:
		raise ValueError(f'Unknown label: {label}')


def format_predictions(preds, labels):
	values = [None for _ in range(len(labels))]
	for l_name, l_value in preds.items():
		label_idx = labels[l_name]
		values[label_idx] = l_value
	return values


def flatten(l):
	return [item for sublist in l for item in sublist]


def filter_tweet_text(tweet_text):
	# TODO consider @<user> and <url> replacing users and urls in tweets
	return tweet_text


def align_tokens(tokens, wpt_tokens, seq_offset=0):
	align_map = {}
	for token in tokens:
		token['wpt_idxs'] = set()
		start = token['start']
		end = token['end']
		for char_idx in range(start, end):
			sub_token_idx = wpt_tokens.char_to_token(char_idx, sequence_index=seq_offset)
			# White spaces have no token and will return None
			if sub_token_idx is not None:
				align_map[sub_token_idx] = token
				token['wpt_idxs'].add(sub_token_idx)
	return align_map


def align_token_sequences(m_tokens, t_tokens, wpt_tokens):
	# print([f'{i}:{m}' for i, m in enumerate(wpt_tokens.tokens())])
	# print([f'{m["start"]}:{m["end"]}:{m["text"]}' for m in m_tokens])
	# print([f'{m["start"]}:{m["end"]}:{m["text"]}' for m in t_tokens])
	m_align_map = align_tokens(m_tokens, wpt_tokens)
	t_align_map = align_tokens(t_tokens, wpt_tokens, seq_offset=1)
	align_map = {**m_align_map, **t_align_map}
	aligned_tokens = []
	for sub_token_idx in range(len(wpt_tokens['input_ids'])):
		if sub_token_idx not in align_map:
			# CLS, SEP, or other special token
			aligned_token = {
				'pos': 'NONE',
				'dep': 'NONE',
				'head': 'NONE',
				'sentic': None,
				'text': '[CLS]' if sub_token_idx == 0 else '[SEP]',
				'wpt_idxs': {sub_token_idx}
			}
			align_map[sub_token_idx] = aligned_token
		aligned_token = align_map[sub_token_idx]
		aligned_tokens.append(aligned_token)

	return align_map, aligned_tokens


def create_adjacency_matrix(edges, size, t_map, r_map):
	adj = np.eye(size, dtype=np.float32)
	for input_idx in range(size):
		input_idx_text = t_map[input_idx]
		i_edges = set(flatten([r_map[e_txt] for e_txt in edges[input_idx_text]]))
		for edge_idx in i_edges:
			adj[input_idx, edge_idx] = 1.0
			adj[edge_idx, input_idx] = 1.0
	return adj


def create_edges(m_tokens, t_tokens, wpt_tokens, lex_edge_expanded):
	seq_len = len(wpt_tokens['input_ids'])
	align_map, a_tokens = align_token_sequences(m_tokens, t_tokens, wpt_tokens)

	lexical_edges = defaultdict(set)
	reverse_lexical_dep_edges = defaultdict(set)
	reverse_lexical_pos_edges = defaultdict(set)
	lexical_dep_edges = defaultdict(set)
	lexical_pos_edges = defaultdict(set)
	root_text = None
	r_map = defaultdict(set)
	t_map = {}
	for token in a_tokens:
		text = token['text'].lower()
		head = token['head'].lower()
		for wpt_idx in token['wpt_idxs']:
			t_map[wpt_idx] = text
			r_map[text].add(wpt_idx)
		pos = token['pos']
		dep = token['dep']
		reverse_lexical_dep_edges[dep].add(text)
		reverse_lexical_pos_edges[pos].add(text)
		lexical_dep_edges[text].add(dep)
		lexical_pos_edges[text].add(pos)
		# will be two roots with two sequences
		if dep == 'ROOT':
			root_text = text
		lexical_edges[text].add(head)

	lexical_edges['[CLS]'].add(root_text)
	lexical_edges['[SEP]'].add(root_text)

	if 'dep' in lex_edge_expanded:
		for text in lexical_edges.keys():
			# expand lexical edges to same dependency roles
			text_deps = lexical_dep_edges[text]
			lexical_edges[text] = lexical_edges[text].union(
				set(flatten(reverse_lexical_dep_edges[dep] for dep in text_deps))
			)

	if 'pos' in lex_edge_expanded:
		for text in lexical_edges.keys():
			# expand lexical edges to same pos tags
			text_pos = lexical_pos_edges[text]
			lexical_edges[text] = lexical_edges[text].union(
				set(flatten(reverse_lexical_pos_edges[pos] for pos in text_pos))
			)

	lexical_adj = create_adjacency_matrix(
		edges=lexical_edges,
		size=seq_len,
		t_map=t_map,
		r_map=r_map
	)

	edges = {
		'lexical': lexical_adj,
	}

	return edges


def get_token_features(token):
	token_data = {
		'text': token.text,
		'pos': token.pos_,
		'dep': token.dep_,
		'head': token.head.text,
		'start': token.idx,
		'end': token.idx + len(token.text),
	}
	return token_data


class MisinfoBatchSampler(Sampler):
	def __init__(self, dataset, pos_count: int, neg_count: int = 0, epoch_shuffle=True, seed=0):
		super().__init__(dataset)
		self.dataset = dataset
		self.generator = None
		self.epoch_shuffle = epoch_shuffle
		self.seed = seed
		self.m_pos_examples = defaultdict(list)
		self.m_neg_examples = defaultdict(list)
		self.m_ids = {}
		for ex_idx in range(len(dataset)):
			ex = dataset[ex_idx]
			for m_id in ex['m_pos_labels']:
				self.m_pos_examples[m_id].append(ex_idx)
				if m_id not in self.m_ids:
					self.m_ids[m_id] = len(self.m_ids)
			for m_id in ex['m_neg_labels']:
				self.m_neg_examples[m_id].append(ex_idx)
				if m_id not in self.m_ids:
					self.m_ids[m_id] = len(self.m_ids)

		assert pos_count <= len(self.m_ids)
		self.m_ids_list = list(self.m_ids.keys())
		self.m_idxs = {m_idx: m_id for m_id, m_idx in self.m_ids.items()}
		self.pos_count = pos_count
		self.neg_count = neg_count
		self.batch_size = self.pos_count + self.neg_count
		self.total_pos_count = sum([len(x) for x in self.m_pos_examples.values()])
		self.total_neg_count = sum([len(x) for x in self.m_neg_examples.values()])

	def __iter__(self):
		# create new generator if this is the first time, otherwise re-create same generator with same seed
		# every time if we do not want to shuffle every epoch.
		if self.generator is None or not self.epoch_shuffle:
			self.generator = torch.Generator()
			self.generator.manual_seed(self.seed)

		batch = []
		num_batches = self.total_pos_count // self.batch_size
		for b_idx in range(num_batches):
			m_idxs = self.sample_misinfo(self.pos_count, self.generator)
			for m_idx in m_idxs:
				m_id = self.m_idxs[m_idx]
				ex_idx = self.sample_positive(m_id, self.generator)
				batch.append(ex_idx)
			yield batch
			batch = []

	def __len__(self):
		return self.total_pos_count // self.batch_size

	def sample_misinfo(self, m_count, generator):
		m_s_indices = torch.randperm(
			n=len(self.m_ids),
			generator=generator
		).tolist()[:m_count]
		return m_s_indices

	def sample_positive(self, m_id, generator):
		pos_examples = self.m_pos_examples[m_id]
		s_idx = torch.randint(
			high=len(pos_examples),
			size=(1,),
			dtype=torch.int64,
			generator=generator
		).tolist()[0]
		ex_idx = pos_examples[s_idx]
		return ex_idx

	def sample_negative(self, m_id, generator):
		neg_examples = self.m_neg_examples[m_id]
		s_idx = torch.randint(
			high=len(neg_examples),
			size=(1,),
			dtype=torch.int64,
			generator=generator
		).tolist()[0]
		ex_idx = neg_examples[s_idx]
		return ex_idx


class MisinfoDataset(Dataset):
	def __init__(
			self,
			documents,
			tokenizer,
	):
		self.examples = []
		self.num_labels = defaultdict(int)
		self.num_classes = defaultdict(int)

		for doc in tqdm(documents, desc='loading documents...'):
			tweet_id = doc['id']
			tweet_text = doc['full_text'].strip().replace('\r', ' ').replace('\n', ' ')
			tweet_text = filter_tweet_text(tweet_text)
			# if create_edge_features:
			# 	tweet_parse = [get_token_features(x) for x in nlp(tweet_text)]
			token_data = tokenizer(
				tweet_text
			)
			labels = {}
			m_pos_labels = set()
			m_neg_labels = set()
			for m_id, m_label in doc['misinfo'].items():
				m_label = label_text_to_relevant_id(m_label)
				labels[m_id] = m_label
				if m_label > 0:
					m_pos_labels.add(m_id)
				else:
					m_neg_labels.add(m_id)
				self.num_labels[m_label] += 1
				self.num_classes[m_id] += 1

			ex = {
				'id': tweet_id,
				'text': tweet_text,
				'input_ids': token_data['input_ids'],
				'token_type_ids': token_data['token_type_ids'],
				'attention_mask': token_data['attention_mask'],
				'labels': labels,
				'm_pos_labels': m_pos_labels,
				'm_neg_labels': m_neg_labels,
			}

			self.examples.append(ex)

		random.shuffle(self.examples)
		self.num_examples = len(self.examples)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example


class MisinfoBatchCollator(object):
	def __init__(
			self, max_seq_len: int, force_max_seq_len: bool, misinfo: dict, tokenizer,
			labeled=True):
		super().__init__()
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len
		# {m_id -> {title, text, alternate_text, source}}
		self.misinfo = misinfo
		self.tokenizer = tokenizer
		for m_id, m in self.misinfo.items():
			m['token_data'] = tokenizer(
				m['text']
			)
		self.labeled = labeled

	def __call__(self, examples):
		ids = []
		# [labels..., tweets...]
		pad_seq_len = 0

		for ex in examples:
			pad_seq_len = max(pad_seq_len, min(len(ex['input_ids']), self.max_seq_len))
		# TODO if force_max_seq_len then m_idx_map should have all m_ids in it to maintain same batch size or mask
		m_idx_map = {}
		# m_idx_map = {m_id: m_idx for (m_idx, m_id) in enumerate(self.misinfo)}
		for ex_idx, ex in enumerate(examples):
			ids.append(ex['id'])
			for m_id in ex['m_pos_labels']:
				if m_id not in m_idx_map:
					m_idx_map[m_id] = len(m_idx_map)

		for m_id, m_idx in m_idx_map.items():
			m = self.misinfo[m_id]
			pad_seq_len = max(pad_seq_len, min(len(m['token_data']['input_ids']), self.max_seq_len))

		batch_size = len(m_idx_map) + len(examples)
		input_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)

		# [ex_count, m_count]
		labels = torch.zeros([len(examples), len(m_idx_map)], dtype=torch.long)
		if self.labeled:
			for ex_idx, ex in enumerate(examples):
				for m_id in ex['m_pos_labels']:
					m_idx = m_idx_map[m_id]
					labels[ex_idx, m_idx] = 1

		# [1, m_count]
		ex_pos_count = labels.eq(1).long().sum(dim=0, keepdim=True).float()

		# [ex_count, 1]
		m_pos_count = labels.eq(1).long().sum(dim=1, keepdim=True).float()

		# [1, m_count]
		ex_mask = (labels.eq(0).long()).sum(dim=0, keepdim=True).gt(0)
		# [ex_count, 1]
		m_mask = (labels.eq(0).long()).sum(dim=1, keepdim=True).gt(0)
		# [ex_count, m_count]
		loss_mask = torch.bitwise_and(ex_mask, m_mask).long()

		for m_id, m_idx in m_idx_map.items():
			m = self.misinfo[m_id]
			self.pad_and_apply(m['token_data']['input_ids'], input_ids, m_idx)
			self.pad_and_apply(m['token_data']['attention_mask'], attention_mask, m_idx)
			self.pad_and_apply(m['token_data']['token_type_ids'], token_type_ids, m_idx)

		for ex_idx, ex in enumerate(examples):
			b_idx = len(m_idx_map) + ex_idx
			self.pad_and_apply(ex['input_ids'], input_ids, b_idx)
			self.pad_and_apply(ex['attention_mask'], attention_mask, b_idx)
			self.pad_and_apply(ex['token_type_ids'], token_type_ids, b_idx)

		batch = {
			'id': ids,
			'm_ids': list(m_idx_map.values()),
			'num_misinfo': len(m_idx_map),
			'num_examples': len(examples),
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'labels': labels,
			'ex_pos_count': ex_pos_count,
			'm_pos_count': m_pos_count,
			'ex_mask': ex_mask,
			'm_mask': m_mask,
			'loss_mask': loss_mask
		}

		return batch

	def pad_and_apply(self, id_list, id_tensor, ex_idx):
		ex_ids = id_list[:self.max_seq_len]
		id_tensor[ex_idx, :len(ex_ids)] = torch.tensor(ex_ids, dtype=torch.long)
