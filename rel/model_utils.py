
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


class BaseCovidTwitterMisinfoModel(pl.LightningModule):
	def __init__(
			self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total, threshold=None,
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
		self.save_hyperparameters()
		self.batch_log = {}

	def _forward_step(self, batch):
		t_ex_embs, m_embs, pos_energy, neg_energy, rel_energy = self(batch)
		scores = -rel_energy
		if not self.predict_mode:
			loss, accuracy = self.loss(pos_energy, neg_energy, batch['subj_obj_mask'])

			return loss, accuracy, scores
		else:
			return t_ex_embs, m_embs, scores

	def training_step(self, batch, batch_nb):
		loss, accuracy, scores = self._forward_step(batch)
		self.log('train_loss', loss)
		self.log('train_accuracy', accuracy)
		for log_name, log_value in self.batch_log.items():
			self.log(f'train_{log_name}', log_value)
		result = {
			'loss': loss
		}
		return result

	def test_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'test')

	def validation_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'val')

	def _eval_step(self, batch, batch_nb, name):
		if not self.predict_mode:
			loss, accuracy, scores = self._forward_step(batch)
			loss = loss.detach()
			accuracy = accuracy.detach()
			result = {
				f'{name}_loss': loss,
				f'{name}_batch_loss': loss,
				f'{name}_batch_accuracy': accuracy,
				# f'{name}_batch_scores': scores.detach(),
				# f'{name}_batch_labels': batch['labels'].detach(),
			}

			return result
		else:
			# TODO
			raise NotImplementedError()
			ex_embs, m_embs, scores = self._forward_step(batch)
			scores = scores.detach()
			device_id = get_device_id()
			if len(scores.shape) == 2:
				ids = []
				m_ids = []
				m_scores = []

				for b_idx, b_id in enumerate(batch['id']):
					for m_idx, m_id in enumerate(batch['m_ids']):
						m_score = scores[b_idx, m_idx].item()
						ids.append(b_id)
						m_ids.append(m_id)
						m_scores.append(m_score)

				ex_dict = {
					'id': ids,
					'm_id': m_ids,
					'm_score': m_scores,
				}
			else:
				ex_dict = {
					'id': batch['id'],
					'm_id': batch['m_ids'],
					'm_score': scores.tolist()
				}
			self.write_prediction_dict(
				ex_dict,
				filename=os.path.join(self.predict_path, f'predictions-{device_id}.pt')
			)
			result = {
				f'{name}_id': batch['id'],
				f'{name}_m_ids': batch['m_ids'],
				f'{name}_scores': scores,
			}

			return result

	def _get_predictions(self, scores, threshold):
		# normalize over
		predictions = (scores.gt(threshold)).long()
		return predictions

	def _get_metrics(self, scores, labels, threshold, name):
		metrics = {}
		# [num_examples, num_misinfo]
		predictions = self._get_predictions(scores, threshold)
		# label is positive and predicted positive
		i_tp = (predictions.eq(1).float() * labels.eq(1).float()).sum()
		# label is not positive and predicted positive
		i_fp = (predictions.eq(1).float() * labels.ne(1).float()).sum()
		# label is positive and predicted negative
		i_fn = (predictions.ne(1).float() * labels.eq(1).float()).sum()
		i_precision = i_tp / (torch.clamp(i_tp + i_fp, 1.0))
		i_recall = i_tp / torch.clamp(i_tp + i_fn, 1.0)

		i_f1 = 2.0 * (i_precision * i_recall) / (torch.clamp(i_precision + i_recall, 1.0))
		metrics[f'{name}_f1'] = i_f1
		metrics[f'{name}_p'] = i_precision
		metrics[f'{name}_r'] = i_recall
		metrics[f'{name}_threshold'] = threshold
		return metrics

	def _eval_epoch_end(self, outputs, name):
		if not self.predict_mode:
			# TODO
			# scores = torch.cat([x[f'{name}_batch_scores'].flatten() for x in outputs], dim=0)
			# labels = torch.cat([x[f'{name}_batch_labels'].flatten() for x in outputs], dim=0)
			#
			# if self.threshold is None:
			# 	# cosine similarities between -1 and 1
			# 	threshold_range = self._get_threshold_range()
			# else:
			# 	threshold_range = [self.threshold]
			# max_metric = float('-inf')
			# max_metrics = {}
			# for threshold in threshold_range:
			# 	t_metrics = self._get_metrics(scores, labels, threshold, name)
			# 	m = t_metrics[f'{name}_f1']
			# 	if m > max_metric:
			# 		max_metric = m
			# 		max_metrics = t_metrics
			#
			# for metric, value in max_metrics.items():
			# 	self.log(metric, value)

			loss = torch.cat([x[f'{name}_batch_loss'].flatten() for x in outputs], dim=0)
			accuracy = torch.cat([x[f'{name}_batch_accuracy'].flatten() for x in outputs], dim=0)
			loss = loss.mean()
			accuracy = accuracy.mean()
			self.log(f'{name}_loss', loss)
			self.log(f'{name}_accuracy', accuracy)

	def validation_epoch_end(self, outputs):
		self._eval_epoch_end(outputs, 'val')

	def test_epoch_end(self, outputs):
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

	def _get_threshold_range(self):
		return np.arange(
			start=-10.00,
			stop=1.00,
			step=0.01
		)


class CovidTwitterMisinfoModel(BaseCovidTwitterMisinfoModel):
	def __init__(
			self, emb_model, emb_size, *args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.emb_size = emb_size

		self.f_dropout = nn.Dropout(
			p=self.config.hidden_dropout_prob
		)
		# TODO determine from emb_model
		self.emb_model = TransDEmbedding(
			self.config.hidden_size,
			self.emb_size
		)

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
		ex_embs, ex_projs = self.emb_model(lm_output)
		# [bsize, num_seq, emb_size]
		ex_embs = ex_embs.view(num_examples, num_sequences_per_example, self.emb_size)
		# [bsize, num_seq, emb_size]
		ex_projs = ex_projs.view(num_examples, num_sequences_per_example, self.emb_size)

		# all [bsize, emb_size], [bsize, emb_size], [bsize, pos_samples, emb_size], [bsize, neg_samples, emb_size]
		t_ex_embs, m_embs, pos_embs, neg_embs = self._split_embeddings(ex_embs, batch)
		t_ex_projs, m_projs, pos_projs, neg_projs = self._split_embeddings(ex_projs, batch)
		# [bsize, 1, emb_size]
		t_ex_embs = t_ex_embs.unsqueeze(dim=-2)
		t_ex_projs = t_ex_projs.unsqueeze(dim=-2)
		m_embs = m_embs.unsqueeze(dim=-2)
		m_projs = m_projs.unsqueeze(dim=-2)
		t_ex_embs = t_ex_embs, t_ex_projs
		m_embs = m_embs, m_projs
		pos_embs = pos_embs, pos_projs
		neg_embs = neg_embs, neg_projs

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
		# [bsize]
		rel_energy = self.emb_model.rel_energy(t_ex_embs, m_embs)

		return t_ex_embs, m_embs, pos_energy, neg_energy, rel_energy

	def _split_embeddings(self, embs, batch):
		t_ex_embs = embs[:, 0]
		m_embs = embs[:, 1]
		pos_samples = batch['pos_samples']
		pos_embs = embs[:, 2:2+pos_samples]
		neg_samples = batch['neg_samples']
		neg_embs = embs[:, 2+pos_samples:2+pos_samples+neg_samples]

		return t_ex_embs, m_embs, pos_embs, neg_embs

	def loss(self, pos_energy, neg_energy, subj_obj_mask):
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
		self.e_emb_layer = nn.Linear(
			hidden_size,
			self.emb_size
		)
		self.e_proj_layer = nn.Linear(
			hidden_size,
			self.emb_size
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

		return ex_embs, ex_projs

	def project(self, c, c_proj, r_proj):
		c_p = c + torch.sum(c * c_proj, dim=-1, keepdim=True) * r_proj
		c_p_norm = torch.norm(c_p, p=2, dim=-1, keepdim=True)
		c_p = c_p / torch.clamp(c_p_norm, max=1.0)
		return c_p

	def energy(self, head, rel, tail):
		h, h_proj = head
		r, r_proj = rel
		t, t_proj = tail
		h_p = self.project(h, h_proj, r_proj)
		t_p = self.project(t, t_proj, r_proj)
		h_r_t_diff = h_p + r - t_p
		h_r_t_energy = torch.norm(h_r_t_diff, p=2, dim=-1, keepdim=False)
		return h_r_t_energy

	def rel_energy(self, head, rel):
		h, h_proj = head
		r, r_proj = rel
		h_p = self.project(h, h_proj, r_proj)
		h_r_t_diff = r - h_p
		h_r_t_energy = torch.norm(h_r_t_diff, p=2, dim=-1, keepdim=False)
		return h_r_t_energy

