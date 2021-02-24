
import json
import os
import json
import torch
from torch.utils.data import Dataset
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
			# TODO work on this
			token_data = tokenizer(
				tweet_text
			)
			labels = {}
			for m_id, m_label in doc['misinfo'].items():
				self.num_labels[m_label] += 1
				self.num_classes[m_id] += 1
				labels[m_id] = label_text_to_relevant_id(m_label)

			ex = {
				'id': tweet_id,
				'text': tweet_text,
				'input_ids': token_data['input_ids'],
				'token_type_ids': token_data['token_type_ids'],
				'attention_mask': token_data['attention_mask'],
				'labels': labels
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
		self.m_pad_seq_len = 0
		for m_id, m in self.misinfo.items():
			m['token_data'] = tokenizer(
				m['text']
			)
			self.m_pad_seq_len = max(self.m_pad_seq_len, min(len(m['token_data']['input_ids']), self.max_seq_len))
		self.labeled = labeled

	def __call__(self, examples):
		ids = []
		m_ids = []
		# [labels..., tweets...]
		batch_size = len(self.misinfo) + len(examples)
		pad_seq_len = self.m_pad_seq_len

		for ex in examples:
			pad_seq_len = max(pad_seq_len, min(len(ex['input_ids']), self.max_seq_len))

		input_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)

		labels = torch.zeros([len(examples), len(self.misinfo)], dtype=torch.long)

		m_idx_map = {}
		for m_idx, (m_id, m) in enumerate(self.misinfo.items()):
			self.pad_and_apply(m['token_data']['input_ids'], input_ids, m_idx)
			self.pad_and_apply(m['token_data']['attention_mask'], attention_mask, m_idx)
			self.pad_and_apply(m['token_data']['token_type_ids'], token_type_ids, m_idx)
			m_idx_map[m_id] = m_idx

		for ex_idx, ex in enumerate(examples):
			ids.append(ex['id'])
			if self.labeled:
				ex_m_ids = []
				for m_id, m_label in ex['labels'].items():
					m_idx = m_idx_map[m_id]
					labels[ex_idx, m_idx] = m_label
					ex_m_ids.append(m_id)
				m_ids.append(ex_m_ids)
			b_idx = ex_idx + len(self.misinfo)
			self.pad_and_apply(ex['input_ids'], input_ids, b_idx)
			self.pad_and_apply(ex['attention_mask'], attention_mask, b_idx)
			self.pad_and_apply(ex['token_type_ids'], token_type_ids, b_idx)

		batch = {
			'id': ids,
			'm_ids': m_ids,
			'num_misinfo': len(self.misinfo),
			'num_examples': len(examples),
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'labels': labels
		}

		return batch

	def pad_and_apply(self, id_list, id_tensor, ex_idx):
		ex_ids = id_list[:self.max_seq_len]
		id_tensor[ex_idx, :len(ex_ids)] = torch.tensor(ex_ids, dtype=torch.long)