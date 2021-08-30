
import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np


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
	if label == 'Agree':
		return 1
	if label == 'Disagree':
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


def flatten(list_of_lists):
	return [item for sublist in list_of_lists for item in sublist]


def filter_tweet_text(tweet_text):
	# TODO consider @<user> and <url> replacing users and urls in tweets
	return tweet_text


class MisinfoDataset(Dataset):
	def __init__(
			self,
			documents,
			tokenizer,
			misinfo,
			pos_samples=1,
			neg_samples=1,
			shuffle=False,
			neg_labels=False
	):
		self.generator = None
		self.shuffle = shuffle
		self.misinfo = misinfo
		self.neg_samples = neg_samples
		self.pos_samples = pos_samples
		self.neg_labels = neg_labels
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
				d_misinfo = doc['labels']
				if m_id in d_misinfo:
					m_label = label_text_to_relevant_id(d_misinfo[m_id])
					if m_label > 0:
						self.examples.append((t_ex, m_ex, m_label))
						self.pos_examples[m_id].append(t_ex)
					else:
						# "hard" negatives created here
						# TODO consider whether negative samples should be "hard" or "soft"
						# TODO "hard" being annotated negatives, "soft" being sampled from all other tweets
						if self.neg_labels:
							self.examples.append((t_ex, m_ex, m_label))
						self.neg_examples[m_id].append(t_ex)
					self.num_labels[m_label] += 1
					self.num_classes[m_id] += 1
				else:
					# "soft" negatives created here
					self.neg_examples[m_id].append(t_ex)
					if self.neg_labels:
						self.examples.append((t_ex, m_ex, 0))

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

		t_ex, m_ex, m_label = self.examples[idx]
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
			'label': m_label,
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
			force_max_seq_len=False
	):
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
		labels = []
		for ex_idx, ex in enumerate(examples):
			ids.append(ex['t_ex']['id'])
			m_ids.append(ex['m_ex']['m_id'])
			labels.append(ex['label'])
			ex_seqs = [ex['m_ex'], ex['t_ex']] + ex['p_samples'] + ex['n_samples']
			subj_obj_mask[ex_idx, ex['subj_obj_sample']] = 1.0
			for seq_idx, seq in enumerate(ex_seqs):
				self.pad_and_apply(seq['token_data']['input_ids'], input_ids, ex_idx, seq_idx)
				self.pad_and_apply(seq['token_data']['attention_mask'], attention_mask, ex_idx, seq_idx)
				self.pad_and_apply(seq['token_data']['token_type_ids'], token_type_ids, ex_idx, seq_idx)
		batch = {
			'ids': ids,
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
			'labels': torch.tensor(labels, dtype=torch.long),
		}

		return batch

	def pad_and_apply(self, id_list, id_tensor, ex_idx, seq_idx):
		ex_ids = id_list[:self.max_seq_len]
		id_tensor[ex_idx, seq_idx, :len(ex_ids)] = torch.tensor(ex_ids, dtype=torch.long)


class MisinfoEntityDataset(Dataset):
	def __init__(
			self,
			documents,
			tokenizer,
			misinfo
	):
		self.m_examples = defaultdict(list)
		self.examples = []
		for doc in tqdm(documents, desc='loading documents...'):
			tweet_id = doc['id']
			tweet_text = doc['full_text'].strip().replace('\r', ' ').replace('\n', ' ')
			tweet_text = filter_tweet_text(tweet_text)

			token_data = tokenizer(
				tweet_text
			)
			t_labels = []
			for m_id, m in misinfo.items():
				d_misinfo = doc['labels']
				if m_id in d_misinfo:
					m_label = label_text_to_relevant_id(d_misinfo[m_id])
					if m_label > 0:
						t_labels.append(m_id)
						self.m_examples[m_id].append(tweet_id)
			ex = {
				'id': tweet_id,
				'e_type': 'entity',
				'token_data': token_data,
				't_labels': t_labels
			}
			self.examples.append(ex)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		return self.examples[idx]


class MisinfoRelDataset(Dataset):
	def __init__(
			self,
			misinfo,
			tokenizer,
			m_examples
	):
		self.examples = []
		for m_id, m in misinfo.items():
			m_token_data = tokenizer(
				m['text']
			)
			m_ex = {
				'id': m_id,
				'e_type': 'rel',
				'm_examples': m_examples[m_id],
				'token_data': m_token_data
			}
			self.examples.append(m_ex)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		return self.examples[idx]


class MisinfoPredictBatchCollator:
	def __init__(
			self, max_seq_len: int,
			force_max_seq_len=False
	):
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def _calculate_seq_padding(self, examples):
		if self.force_max_seq_len:
			pad_seq_len = self.max_seq_len
		else:
			pad_seq_len = 0
			for ex in examples:
				pad_seq_len = max(pad_seq_len, min(len(ex['token_data']['input_ids']), self.max_seq_len))
		return pad_seq_len

	def __call__(self, examples):
		pad_seq_len = self._calculate_seq_padding(examples)
		num_examples = len(examples)

		input_ids = torch.zeros([num_examples, pad_seq_len], dtype=torch.long)
		attention_mask = torch.zeros([num_examples, pad_seq_len], dtype=torch.long)
		token_type_ids = torch.zeros([num_examples, pad_seq_len], dtype=torch.long)
		ids = []
		m_examples = []
		t_labels = []
		for ex_idx, ex in enumerate(examples):
			ids.append(ex['id'])
			if 'm_examples' in ex:
				m_examples.append(','.join(ex['m_examples']))
			if 't_labels' in ex:
				t_labels.append(','.join(ex['t_labels']))
			self.pad_and_apply(ex['token_data']['input_ids'], input_ids, ex_idx)
			self.pad_and_apply(ex['token_data']['attention_mask'], attention_mask, ex_idx)
			self.pad_and_apply(ex['token_data']['token_type_ids'], token_type_ids, ex_idx)

		batch = {
			'ids': ids,
			'e_type': examples[0]['e_type'],
			'num_examples': num_examples,
			'pad_seq_len': pad_seq_len,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids
		}
		if len(m_examples) > 0:
			batch['m_examples'] = m_examples

		if len(t_labels) > 0:
			batch['t_labels'] = t_labels

		return batch

	def pad_and_apply(self, id_list, id_tensor, ex_idx):
		ex_ids = id_list[:self.max_seq_len]
		id_tensor[ex_idx, :len(ex_ids)] = torch.tensor(ex_ids, dtype=torch.long)
