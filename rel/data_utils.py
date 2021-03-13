
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
			misinfo,
			pos_samples=1,
			neg_samples=1,
			shuffle=False
	):
		self.generator = None
		self.shuffle = shuffle
		self.misinfo = misinfo
		self.neg_samples = neg_samples
		self.pos_samples = pos_samples
		for m_id, m in self.misinfo.items():
			m['token_data'] = tokenizer(
				m['text']
			)
		self.examples = []
		self.pos_examples = defaultdict(list)
		self.neg_examples = defaultdict(list)

		self.num_labels = defaultdict(int)
		self.num_classes = defaultdict(int)

		for doc in tqdm(documents, desc='loading documents...'):
			tweet_id = doc['id']
			tweet_text = doc['full_text'].strip().replace('\r', ' ').replace('\n', ' ')
			tweet_text = filter_tweet_text(tweet_text)

			token_data = tokenizer(
				tweet_text
			)
			for m_id, m in self.misinfo.items():
				m_ex = {
					'm_id': m_id,
					'text': m['text'],
					'token_data': m['token_data']
				}
				t_ex = {
					'id': tweet_id,
					'text': tweet_text,
					'token_data': token_data,
				}
				d_misinfo = doc['misinfo']
				if m_id in d_misinfo:
					m_label = label_text_to_relevant_id(d_misinfo[m_id])
					if m_label > 0:
						self.examples.append((t_ex, m_ex))
						self.pos_examples[m_id].append(t_ex)
					else:
						# "hard" negatives created here
						# TODO consider whether negative samples should be "hard" or "soft"
						# TODO "hard" being annotated negatives, "soft" being sampled from all other tweets
						self.neg_examples[m_id].append(t_ex)
					self.num_labels[m_label] += 1
					self.num_classes[m_id] += 1
				else:
					# "soft" negatives created here
					self.neg_examples[m_id].append(t_ex)

	def __len__(self):
		return len(self.examples)

	def worker_init_fn(self, _):
		if self.generator is None or not self.shuffle:
			self.generator = torch.Generator()
			worker_info = torch.utils.data.get_worker_info()
			if worker_info is None:
				self.generator.manual_seed(0)
			else:
				self.generator.manual_seed(worker_info.seed)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		t_ex, m_ex = self.examples[idx]
		m_id = m_ex['m_id']

		pos_samples = self._sample(
			self.pos_examples[m_id],
			self.pos_samples
		)

		neg_samples = self._sample(
			self.neg_examples[m_id],
			self.neg_samples
		)

		subj_obj_sample = self._sample_subj_obj()

		ex = {
			't_ex': t_ex,
			'm_ex': m_ex,
			'p_samples': pos_samples,
			'n_samples': neg_samples,
			'subj_obj_sample': subj_obj_sample
		}

		return ex

	def _sample(self, m_examples, m_count):
		samples = []
		if m_count <= 0:
			return samples
		m_s_indices = torch.randperm(
			n=len(m_examples),
			generator=self.generator
		).tolist()[:m_count]
		for s_idx in m_s_indices:
			samples.append(m_examples[s_idx])
		return samples

	def _sample_subj_obj(self):
		r = torch.rand(
			size=(1,),
			generator=self.generator
		).tolist()[0]
		if r < 0.5:
			return 0
		else:
			return 1


class MisinfoBatchCollator:
	def __init__(
			self, max_seq_len: int,
			force_max_seq_len=False):
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def _calculate_seq_padding(self, examples):
		if self.force_max_seq_len:
			pad_seq_len = self.max_seq_len
		else:
			pad_seq_len = 0
			for ex in examples:
				ex_seqs = [ex['t_ex'], ex['m_ex']] + ex['p_samples'] + ex['n_samples']
				for ex_seq in ex_seqs:
					pad_seq_len = max(pad_seq_len, min(len(ex_seq['token_data']['input_ids']), self.max_seq_len))
		return pad_seq_len

	def __call__(self, examples):
		pad_seq_len = self._calculate_seq_padding(examples)
		num_examples = len(examples)
		pos_samples = len(examples[0]['p_samples'])
		neg_samples = len(examples[0]['n_samples'])
		num_sequences_per_example = 2 + pos_samples + neg_samples
		# ex + m + pos_samples + neg_samples
		num_sequences = num_examples * num_sequences_per_example

		input_ids = torch.zeros([num_examples, num_sequences_per_example, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([num_examples, num_sequences_per_example, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([num_examples, num_sequences_per_example, pad_seq_len], dtype=torch.long)
		subj_obj_mask = torch.zeros([num_examples, 2], dtype=torch.float)
		ids = []
		m_ids = []
		for ex_idx, ex in enumerate(examples):
			ids.append(ex['t_ex']['id'])
			m_ids.append(ex['m_ex']['m_id'])
			ex_seqs = [ex['t_ex'], ex['m_ex']] + ex['p_samples'] + ex['n_samples']
			subj_obj_mask[ex_idx, ex['subj_obj_sample']] = 1.0
			for seq_idx, seq in enumerate(ex_seqs):
				self.pad_and_apply(seq['token_data']['input_ids'], input_ids, ex_idx, seq_idx)
				self.pad_and_apply(seq['token_data']['attention_mask'], attention_mask, ex_idx, seq_idx)
				self.pad_and_apply(seq['token_data']['token_type_ids'], token_type_ids, ex_idx, seq_idx)

		batch = {
			'id': ids,
			'm_ids': m_ids,
			'num_examples': num_examples,
			'pos_samples': pos_samples,
			'neg_samples': neg_samples,
			'pad_seq_len': pad_seq_len,
			'num_sequences_per_example': num_sequences_per_example,
			'num_sequences': num_sequences,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'subj_obj_mask': subj_obj_mask,
		}

		return batch

	def pad_and_apply(self, id_list, id_tensor, ex_idx, seq_idx):
		ex_ids = id_list[:self.max_seq_len]
		id_tensor[ex_idx, seq_idx, :len(ex_ids)] = torch.tensor(ex_ids, dtype=torch.long)

