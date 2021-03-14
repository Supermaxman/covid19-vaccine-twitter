from collections import defaultdict

import pytorch_lightning as pl
from transformers import BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
import torch.distributed as dist
import numpy as np
import os
import math
import logging

import metric_utils


class CovidTwitterMisinfoModel(pl.LightningModule):
	def __init__(
			self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total, emb_model, emb_size, gamma,
			threshold=None,
			torch_cache_dir=None, predict_mode=False, predict_path=None, load_pretrained=False
	):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.torch_cache_dir = torch_cache_dir
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.threshold = threshold
		self.predict_mode = predict_mode
		self.predict_path = predict_path
		if self.predict_mode:
			if not os.path.exists(self.predict_path):
				os.mkdir(self.predict_path)

		self.load_pretrained = load_pretrained
		if self.predict_mode or self.load_pretrained:
			# no need to load pre-trained weights since we will be loading whole model's
			# fine-tuned weights from checkpoint.
			self.config = BertConfig.from_pretrained(
				pre_model_name,
				cache_dir=torch_cache_dir
			)
			self.bert = BertModel(self.config)
		else:
			self.bert = BertModel.from_pretrained(
				pre_model_name,
				cache_dir=torch_cache_dir
			)
			self.config = self.bert.config

		self.emb_size = emb_size
		self.gamma = gamma

		self.f_dropout = nn.Dropout(
			p=self.config.hidden_dropout_prob
		)
		if emb_model == 'transd':
			self.emb_model = TransDEmbedding(
				self.config.hidden_size,
				self.emb_size
			)
		else:
			raise ValueError(f'Unknown embedding model: {emb_model}')

		self.save_hyperparameters()

	def forward(self, batch):
		num_examples = batch['num_examples']
		num_sequences_per_example = batch['num_sequences_per_example']
		pad_seq_len = batch['pad_seq_len']

		# [bsize, num_seq, seq_len] -> [bsize * num_seq, seq_len]
		input_ids = batch['input_ids'].view(num_examples * num_sequences_per_example, pad_seq_len)
		attention_mask = batch['attention_mask'].view(num_examples * num_sequences_per_example, pad_seq_len)
		token_type_ids = batch['token_type_ids'].view(num_examples * num_sequences_per_example, pad_seq_len)

		# [bsize * num_seq, seq_len, hidden_size]
		contextualized_embeddings = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)[0]
		# [bsize * num_seq, hidden_size]
		lm_output = contextualized_embeddings[:, 0]
		lm_output = self.f_dropout(lm_output)
		ex_embs = self.emb_model(lm_output)
		# [bsize, num_seq, emb_size]
		ex_embs = ex_embs.view(num_examples, num_sequences_per_example, self.emb_size)
		return ex_embs

	def _triplet_energy(self, ex_embs, batch):
		# all [bsize, emb_size], [bsize, emb_size], [bsize, pos_samples, emb_size], [bsize, neg_samples, emb_size]
		t_ex_embs, m_embs, pos_embs, neg_embs = self._split_embeddings(ex_embs, batch)
		# [bsize, 1, emb_size]
		t_ex_embs = t_ex_embs.unsqueeze(dim=-2)
		m_embs = m_embs.unsqueeze(dim=-2)

		# [bsize]
		pos_subj_energy = self.emb_model.energy(t_ex_embs, m_embs, pos_embs)
		# [bsize]
		pos_obj_energy = self.emb_model.energy(pos_embs, m_embs, t_ex_embs)
		# [bsize, 2]
		pos_energy = torch.stack([pos_subj_energy, pos_obj_energy], dim=-1)
		# [bsize]
		neg_subj_energy = self.emb_model.energy(t_ex_embs, m_embs, neg_embs)
		# [bsize]
		neg_obj_energy = self.emb_model.energy(neg_embs, m_embs, t_ex_embs)
		# [bsize, 2]
		neg_energy = torch.stack([neg_subj_energy, neg_obj_energy], dim=-1)

		return pos_energy, neg_energy

	def _split_embeddings(self, embs, batch):
		t_ex_embs = embs[:, 0]
		m_embs = embs[:, 1]
		pos_samples = batch['pos_samples']
		if pos_samples > 0:
			pos_embs = embs[:, 2:2+pos_samples]
		else:
			pos_embs = None
		neg_samples = batch['neg_samples']
		if neg_samples > 0:
			neg_embs = embs[:, 2+pos_samples:2+pos_samples+neg_samples]
		else:
			neg_embs = None
		return t_ex_embs, m_embs, pos_embs, neg_embs

	def _loss(self, pos_energy, neg_energy, subj_obj_mask):
		# first randomly pick between subject and object losses
		# [bsize]
		pos_energy = (pos_energy * subj_obj_mask).sum(dim=-1)
		# [bsize]
		neg_energy = (neg_energy * subj_obj_mask).sum(dim=-1)
		margin = pos_energy - neg_energy
		loss = torch.clamp(self.gamma + margin, min=0.0)
		accuracy = (pos_energy.lt(neg_energy)).float().mean()
		loss = loss.mean()
		return loss, accuracy

	def _triplet_step(self, batch):
		ex_embs = self(batch)
		pos_energy, neg_energy = self._triplet_energy(ex_embs, batch)
		loss, accuracy = self._loss(pos_energy, neg_energy, batch['subj_obj_mask'])
		return loss, accuracy

	def _label_step(self, batch):
		ex_embs = self(batch)
		t_ex_embs = ex_embs[:, 0]
		m_embs = ex_embs[:, 1]
		return t_ex_embs, m_embs

	def training_step(self, batch, batch_nb):
		loss, accuracy = self._triplet_step(batch)
		self.log('train_loss', loss)
		self.log('train_accuracy', accuracy)
		result = {
			'loss': loss
		}
		return result

	def test_step(self, batch, batch_nb):
		if self.predict_mode:
			return self._predict_step(batch, 'test')
		else:
			return self._triplet_eval_step(batch, 'test')

	def validation_step(self, batch, batch_nb, dataloader_idx):
		if self.predict_mode:
			return self._predict_step(batch, 'val')
		else:
			if dataloader_idx == 0:
				return self._triplet_eval_step(batch, 'val')
			elif dataloader_idx == 1:
				return self._label_eval_step(batch, 'val')

	def _predict_step(self, batch, name):
		raise NotImplementedError()
		ex_embs, m_embs = self._forward_step(batch)
		scores = scores.detach()
		device_id = get_device_id()
		if len(scores.shape) == 2:
			ids = []
			m_ids = []
			m_scores = []

			for b_idx, b_id in enumerate(batch['ids']):
				for m_idx, m_id in enumerate(batch['m_ids']):
					m_score = scores[b_idx, m_idx].item()
					ids.append(b_id)
					m_ids.append(m_id)
					m_scores.append(m_score)

			ex_dict = {
				'ids': ids,
				'm_id': m_ids,
				'm_score': m_scores,
			}
		else:
			ex_dict = {
				'ids': batch['ids'],
				'm_id': batch['m_ids'],
				'm_score': scores.tolist()
			}
		self.write_prediction_dict(
			ex_dict,
			filename=os.path.join(self.predict_path, f'predictions-{device_id}.pt')
		)
		result = {
			f'{name}_ids': batch['ids'],
			f'{name}_m_ids': batch['m_ids'],
			f'{name}_scores': scores,
		}

		return result

	def _triplet_eval_step(self, batch, name):
		loss, accuracy = self._triplet_step(batch)
		loss = loss.detach()
		accuracy = accuracy.detach()
		result = {
			f'{name}_loss': loss,
			f'{name}_batch_loss': loss,
			f'{name}_batch_accuracy': accuracy,
		}

		return result

	def _label_eval_step(self, batch, name):
		ex_embs, m_embs = self._label_step(batch)
		ex_embs = ex_embs.detach()
		m_embs = m_embs.detach()
		result = {
			f'{name}_ex_embs': ex_embs,
			f'{name}_m_embs': m_embs,
			f'{name}_labels': batch['labels'].detach(),
			f'{name}_ids': batch['ids'],
			f'{name}_m_ids': batch['m_ids'],
		}

		return result

	def _eval_epoch_end(self, outputs, name):
		if isinstance(outputs, list) and name == 'val':
			triplet_eval_outputs, label_eval_outputs = outputs
			# triplet eval is dataloader_idx 0
			loss = torch.cat([x[f'{name}_batch_loss'].flatten() for x in triplet_eval_outputs], dim=0)
			accuracy = torch.cat([x[f'{name}_batch_accuracy'].flatten() for x in triplet_eval_outputs], dim=0)
			loss = loss.mean()
			accuracy = accuracy.mean()
			self.log(f'{name}_loss', loss)
			self.log(f'{name}_accuracy', accuracy)

			# label eval is dataloader_idx 1
			ex_embs = torch.cat([x[f'{name}_ex_embs'] for x in label_eval_outputs], dim=0)
			m_embs = torch.cat([x[f'{name}_m_embs'] for x in label_eval_outputs], dim=0)
			labels = torch.cat([x[f'{name}_labels'] for x in label_eval_outputs], dim=0)
			ex_ids = [ex_id for x in label_eval_outputs for ex_id in x[f'{name}_ids']]
			m_ids = [m_id for x in label_eval_outputs for m_id in x[f'{name}_m_ids']]
			# collect all positive examples under each misinfo and take average embedding
			m_ex_embs_list = defaultdict(list)
			for ex_emb, m_label, m_id in zip(ex_embs, labels, m_ids):
				if m_label > 0:
					m_ex_embs_list[m_id].append(ex_emb)
			m_ex_avg_embs = {}
			default_m_ex_emb = None
			for m_id, m_exs in m_ex_embs_list.items():
				# average embedding for positive examples for m_id
				# [emb_size]
				if len(m_exs) == 1:
					m_ex_emb = m_exs[0]
				else:
					m_ex_emb = torch.cat(m_exs, dim=0).mean(dim=0)
				m_ex_avg_embs[m_id] = m_ex_emb
				if default_m_ex_emb is None:
					default_m_ex_emb = torch.zeros_like(m_ex_emb)

			# unroll avg embeddings across batch for easier calculation
			m_ex_embs = []
			for ex_emb, m_emb, m_label, ex_id, m_id in zip(ex_embs, m_embs, labels, ex_ids, m_ids):
				if m_id not in m_ex_avg_embs:
					m_ex_emb = default_m_ex_emb
				else:
					m_ex_emb = m_ex_avg_embs[m_id]
				m_ex_embs.append(m_ex_emb)
			# [bsize, emb_size]
			m_ex_embs = torch.stack(m_ex_embs, dim=0)
			# max energy is inf, min energy is 0
			m_ex_energies = self.emb_model.energy(ex_embs, m_embs, m_ex_embs)
			# max score is 0, min score is -inf
			# [bsize]
			scores = -m_ex_energies
			threshold_range = np.arange(
				start=-10.00,
				stop=0.00,
				step=0.01
			)
			f1, p, r, threshold = metric_utils.compute_threshold_f1(
				scores,
				labels,
				self.threshold,
				threshold_range
			)
			self.log(f'{name}_f1', f1)
			self.log(f'{name}_p', p)
			self.log(f'{name}_r', r)
			self.log(f'{name}_threshold', threshold)

	def validation_epoch_end(self, outputs):
		if not self.predict_mode:
			self._eval_epoch_end(outputs, 'val')

	def test_epoch_end(self, outputs):
		if not self.predict_mode:
			self._eval_epoch_end(outputs, 'test')

	def configure_optimizers(self):
		params = self._get_optimizer_params(self.weight_decay)
		optimizer = AdamW(
			params,
			lr=self.learning_rate,
			weight_decay=self.weight_decay,
			correct_bias=False
		)
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=self.lr_warmup * self.updates_total,
			num_training_steps=self.updates_total
		)
		return [optimizer], [scheduler]

	def _get_optimizer_params(self, weight_decay):
		param_optimizer = list(self.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_params = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
			 'weight_decay': weight_decay},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		return optimizer_params


def get_device_id():
	try:
		device_id = dist.get_rank()
	except AssertionError:
		if 'XRT_SHARD_ORDINAL' in os.environ:
			device_id = int(os.environ['XRT_SHARD_ORDINAL'])
		else:
			device_id = 0
	return device_id


class TransDEmbedding(nn.Module):
	def __init__(self, hidden_size, emb_size):
		super().__init__()
		self.emb_size = emb_size
		self.td_emb_size = self.emb_size // 2
		self.e_emb_layer = nn.Linear(
			hidden_size,
			self.td_emb_size
		)
		self.e_proj_layer = nn.Linear(
			hidden_size,
			self.td_emb_size
		)

	def forward(self, source_embeddings):
		# [bsize * num_seq, emb_size]
		ex_embs = self.e_emb_layer(source_embeddings)
		# https://www.aclweb.org/anthology/P15-1067.pdf
		# normalize all lookups to max l2 norm of 1
		ex_emb_norms = torch.norm(ex_embs, p=2, dim=-1, keepdim=True)
		# [bsize * num_seq, emb_size]
		ex_embs = ex_embs / torch.clamp(ex_emb_norms, max=1.0)

		ex_projs = self.e_proj_layer(source_embeddings)
		ex_embs = torch.cat([ex_embs, ex_projs], dim=-1)
		return ex_embs

	def project(self, c, c_proj, r_proj):
		c_p = c + torch.sum(c * c_proj, dim=-1, keepdim=True) * r_proj
		c_p_norm = torch.norm(c_p, p=2, dim=-1, keepdim=True)
		c_p = c_p / torch.clamp(c_p_norm, max=1.0)
		return c_p

	def energy(self, head, rel, tail):
		h, h_proj = head[..., :self.td_emb_size], head[..., self.td_emb_size:]
		r, r_proj = rel[..., :self.td_emb_size], rel[..., self.td_emb_size:]
		t, t_proj = tail[..., :self.td_emb_size], tail[..., self.td_emb_size:]
		h_p = self.project(h, h_proj, r_proj)
		t_p = self.project(t, t_proj, r_proj)
		h_r_t_diff = h_p + r - t_p
		h_r_t_energy = torch.norm(h_r_t_diff, p=2, dim=-1, keepdim=False)
		return h_r_t_energy


