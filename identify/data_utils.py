
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


def label_text_to_relevant_id(label):
	# 'agree', 'disagree', 'no_stance', 'not_relevant',
	if label == 'Not Relevant':
		return 0
	if label == 'Accept':
		return 1
	if label == 'Reject':
		return 1
	if label == 'No Stance':
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
			misinfo
	):
		self.examples = []
		self.pos_examples = []
		self.neg_examples = []
		self.num_labels = defaultdict(int)
		self.num_classes = defaultdict(int)

		for doc in tqdm(documents, desc='loading documents...'):
			tweet_id = doc['id']
			tweet_text = doc['text'].strip().replace('\r', ' ').replace('\n', ' ')
			tweet_text = filter_tweet_text(tweet_text)
			# if create_edge_features:
			# 	tweet_parse = [get_token_features(x) for x in nlp(tweet_text)]
			token_data = tokenizer(
				tweet_text
			)
			labels = {}
			m_pos_labels = set()
			m_neg_labels = set()
			for m_id, m_label in doc['labels'].items():
				if m_id not in misinfo:
					continue
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
			if len(m_pos_labels) > 0:
				self.pos_examples.append(ex)
			if len(m_neg_labels) > 0:
				self.neg_examples.append(ex)

		random.shuffle(self.examples)
		random.shuffle(self.pos_examples)
		random.shuffle(self.neg_examples)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.examples[idx]

		return example


class MisinfoPositiveDataset(MisinfoDataset):
	def __init__(
			self,
			documents,
			tokenizer,
			misinfo
	):
		super().__init__(documents, tokenizer, misinfo)

	def __len__(self):
		return len(self.pos_examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.pos_examples[idx]

		return example


class MisinfoPairwiseDataset(MisinfoDataset):
	def __init__(
			self,
			documents,
			tokenizer,
			misinfo,
			all_misinfo=False
	):
		super().__init__(documents, tokenizer, misinfo)
		self.misinfo = misinfo
		self.all_misinfo = all_misinfo
		self.pairwise_examples = []
		self.tokenizer = tokenizer
		for ex in self.examples:
			if all_misinfo:
				misinfo_ids = set(misinfo.keys())
			else:
				misinfo_ids = (ex['m_pos_labels'].union(ex['m_neg_labels']))
			for m_id in misinfo_ids:
				if m_id not in misinfo:
					continue
				m_label = 0
				if m_id in ex['m_pos_labels']:
					m_label = 1
				p_ex = self._create_example(ex, m_id, m_label)
				self.pairwise_examples.append(p_ex)

		random.shuffle(self.pairwise_examples)

	def _create_example(self, ex, m_id, m_label):
		m = self.misinfo[m_id]
		m_text = m['text']
		p_ex = ex.copy()
		p_ex['m_id'] = m_id
		p_ex['labels'] = m_label
		token_data = self.tokenizer(
			m_text,
			ex['text']
		)
		p_ex['input_ids'] = token_data['input_ids']
		p_ex['token_type_ids'] = token_data['token_type_ids']
		p_ex['attention_mask'] = token_data['attention_mask']
		return p_ex

	def __len__(self):
		return len(self.pairwise_examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.pairwise_examples[idx]

		return example


class MisinfoPairwiseEmbDataset(MisinfoDataset):
	def __init__(
			self,
			documents,
			tokenizer,
			misinfo,
			all_misinfo=False
	):
		super().__init__(documents, tokenizer, misinfo)
		self.misinfo = misinfo
		self.all_misinfo = all_misinfo
		self.pairwise_examples = []
		self.tokenizer = tokenizer
		for ex in self.examples:
			if all_misinfo:
				misinfo_ids = set(misinfo.keys())
			else:
				misinfo_ids = (ex['m_pos_labels'].union(ex['m_neg_labels']))
			for m_id in misinfo_ids:
				if m_id not in misinfo:
					continue
				m_label = 0
				if m_id in ex['m_pos_labels']:
					m_label = 1
				p_ex = self._create_example(ex, m_id, m_label)
				self.pairwise_examples.append(p_ex)

		random.shuffle(self.pairwise_examples)

	def _create_example(self, ex, m_id, m_label):
		p_ex = ex.copy()
		p_ex['m_id'] = m_id
		p_ex['labels'] = m_label
		return p_ex

	def __len__(self):
		return len(self.pairwise_examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		example = self.pairwise_examples[idx]

		return example


class MisinfoBatchCollator:
	def __init__(
			self, misinfo: dict, tokenizer, max_seq_len: int,
			labeled=True, all_misinfo=False, neg_misinfo=False, force_max_seq_len=False):
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len
		# {m_id -> {title, text, alternate_text, source}}
		self.misinfo = misinfo
		self.all_misinfo = all_misinfo
		self.neg_misinfo = neg_misinfo
		self.tokenizer = tokenizer
		for m_id, m in self.misinfo.items():
			m['token_data'] = tokenizer(
				m['text']
			)
		self.labeled = labeled

	def _calculate_seq_padding(self, examples, batch_misinfo=None):
		if self.force_max_seq_len:
			pad_seq_len = self.max_seq_len
		else:
			pad_seq_len = 0
			for ex in examples:
				pad_seq_len = max(pad_seq_len, min(len(ex['input_ids']), self.max_seq_len))
			if batch_misinfo is not None:
				for m_id, m_idx in batch_misinfo.items():
					m = self.misinfo[m_id]
					pad_seq_len = max(pad_seq_len, min(len(m['token_data']['input_ids']), self.max_seq_len))

		return pad_seq_len

	def _build_batch_misinfo(self, examples):
		# if force_max_seq_len then batch_misinfo should have all m_ids in it to maintain same shape
		if self.all_misinfo or self.force_max_seq_len:
			batch_misinfo = {m_id: m_idx for (m_idx, m_id) in enumerate(self.misinfo)}
		else:
			batch_misinfo = {}
			for ex_idx, ex in enumerate(examples):
				for m_id in ex['m_pos_labels']:
					if m_id not in batch_misinfo:
						batch_misinfo[m_id] = len(batch_misinfo)
				# TODO consider adding m_neg_labels
				if self.neg_misinfo:
					for m_id in ex['m_neg_labels']:
						if m_id not in batch_misinfo:
							batch_misinfo[m_id] = len(batch_misinfo)
		return batch_misinfo

	def __call__(self, examples):
		batch_misinfo = self._build_batch_misinfo(examples)

		pad_seq_len = self._calculate_seq_padding(examples, batch_misinfo)

		batch_size = len(batch_misinfo) + len(examples)
		input_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)

		# [ex_count, m_count]
		labels = torch.zeros([len(examples), len(batch_misinfo)], dtype=torch.long)
		if self.labeled:
			for ex_idx, ex in enumerate(examples):
				for m_id in ex['m_pos_labels']:
					m_idx = batch_misinfo[m_id]
					labels[ex_idx, m_idx] = 1

		for m_id, m_idx in batch_misinfo.items():
			m = self.misinfo[m_id]
			self.pad_and_apply(m['token_data']['input_ids'], input_ids, m_idx)
			self.pad_and_apply(m['token_data']['attention_mask'], attention_mask, m_idx)
			self.pad_and_apply(m['token_data']['token_type_ids'], token_type_ids, m_idx)

		ids = []
		for ex_idx, ex in enumerate(examples):
			ids.append(ex['id'])
			b_idx = len(batch_misinfo) + ex_idx
			self.pad_and_apply(ex['input_ids'], input_ids, b_idx)
			self.pad_and_apply(ex['attention_mask'], attention_mask, b_idx)
			self.pad_and_apply(ex['token_type_ids'], token_type_ids, b_idx)

		batch = {
			'id': ids,
			'm_ids': list(batch_misinfo.keys()),
			'num_misinfo': len(batch_misinfo),
			'num_examples': len(examples),
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'labels': labels,
		}

		return batch

	def pad_and_apply(self, id_list, id_tensor, ex_idx):
		ex_ids = id_list[:self.max_seq_len]
		id_tensor[ex_idx, :len(ex_ids)] = torch.tensor(ex_ids, dtype=torch.long)


class MisinfoPairwiseBatchCollator(MisinfoBatchCollator):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, examples):
		pad_seq_len = self._calculate_seq_padding(examples)

		batch_size = len(examples)
		input_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)

		# [ex_count]
		labels = torch.zeros([len(examples)], dtype=torch.long)
		if self.labeled:
			for ex_idx, ex in enumerate(examples):
				labels[ex_idx] = ex['labels']

		ids = []
		m_ids = []
		for ex_idx, ex in enumerate(examples):
			ids.append(ex['id'])
			m_ids.append(ex['m_id'])
			self.pad_and_apply(ex['input_ids'], input_ids, ex_idx)
			self.pad_and_apply(ex['attention_mask'], attention_mask, ex_idx)
			self.pad_and_apply(ex['token_type_ids'], token_type_ids, ex_idx)

		batch = {
			'id': ids,
			'm_ids': m_ids,
			'num_misinfo': 0,
			'num_examples': len(examples),
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'labels': labels,
		}

		return batch


class MisinfoPairwiseEmbBatchCollator(MisinfoBatchCollator):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, examples):
		batch_misinfo = {ex['m_id']: m_idx for (m_idx, ex) in enumerate(examples)}
		pad_seq_len = self._calculate_seq_padding(examples, batch_misinfo)

		batch_size = len(examples)
		input_ids = torch.zeros([2 * batch_size, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([2 * batch_size, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([2 * batch_size, pad_seq_len], dtype=torch.long)

		# [ex_count]
		labels = torch.zeros([batch_size], dtype=torch.long)
		if self.labeled:
			for ex_idx, ex in enumerate(examples):
				labels[ex_idx] = ex['labels']

		ids = []
		m_ids = []
		for ex_idx, ex in enumerate(examples):
			ids.append(ex['id'])
			m_ids.append(ex['m_id'])
			m_id = ex['m_id']
			m = self.misinfo[m_id]
			self.pad_and_apply(m['token_data']['input_ids'], input_ids, ex_idx)
			self.pad_and_apply(m['token_data']['attention_mask'], attention_mask, ex_idx)
			self.pad_and_apply(m['token_data']['token_type_ids'], token_type_ids, ex_idx)
			b_idx = batch_size + ex_idx
			self.pad_and_apply(ex['input_ids'], input_ids, b_idx)
			self.pad_and_apply(ex['attention_mask'], attention_mask, b_idx)
			self.pad_and_apply(ex['token_type_ids'], token_type_ids, b_idx)

		batch = {
			'id': ids,
			'm_ids': m_ids,
			'num_misinfo': batch_size,
			'num_examples': batch_size,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'labels': labels,
		}

		return batch
