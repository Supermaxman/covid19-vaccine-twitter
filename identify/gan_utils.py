
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


class CovidTwitterPairwiseGenerator(nn.Module):
	def __init__(self, hidden_size, hidden_dropout_prob):
		super().__init__()
		self.cls_classification_layer = nn.Linear(
			hidden_size,
			1
		)
		self.f_dropout = nn.Dropout(
			p=hidden_dropout_prob
		)
		self.score_func = torch.nn.Sigmoid()
		self.loss_metric = torch.nn.BCEWithLogitsLoss(reduction='none')

	def forward(self, contextualized_embeddings):
		# [bsize, seq_len, hidden_size]
		# outputs = self.bert(
		# 	batch['input_ids'],
		# 	attention_mask=batch['attention_mask'],
		# 	token_type_ids=batch['token_type_ids']
		# )
		# contextualized_embeddings = outputs[0]
		# [bsize, hidden_size]
		lm_output = contextualized_embeddings[:, 0]
		lm_output = self.f_dropout(lm_output)
		# [bsize]
		logits = self.cls_classification_layer(lm_output).squeeze(dim=-1)
		scores = self.score_func(logits)

		return logits, scores

	def loss(self, logits, labels):
		return self.loss_metric(
			logits,
			labels.float()
		)


class CovidTwitterPairwiseDiscriminator(nn.Module):
	def __init__(self, hidden_size, hidden_dropout_prob):
		super().__init__()
		self.cls_classification_layer = nn.Linear(
			hidden_size,
			1
		)
		self.f_dropout = nn.Dropout(
			p=hidden_dropout_prob
		)
		self.score_func = torch.nn.Softmax(dim=-1)

	def forward(self, contextualized_embeddings):
		# [bsize, seq_len, hidden_size]
		# outputs = self.bert(
		# 	batch['input_ids'],
		# 	attention_mask=batch['attention_mask'],
		# 	token_type_ids=batch['token_type_ids']
		# )
		# contextualized_embeddings = outputs[0]
		# [bsize, hidden_size]
		lm_output = contextualized_embeddings[:, 0]
		lm_output = self.f_dropout(lm_output)
		# [bsize]
		logits = self.cls_classification_layer(lm_output).squeeze(dim=-1)
		# [bsize]
		scores = self.score_func(logits)

		return logits, scores

	def loss(self, probs, rewards):
		# distribute reward over probability distribution
		# r_rewards = probs.detach() * rewards
		# reinforce each prob based on its proportion of reward
		r_loss = -torch.log(probs + 1e-6) * rewards

		return r_loss


class CovidTwitterPairwiseGanMisinfoModel(pl.LightningModule):
	def __init__(
			self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total, losses, threshold=None,
			torch_cache_dir=None, predict_mode=False, predict_path=None, load_pretrained=False
	):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.torch_cache_dir = torch_cache_dir
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.losses = losses
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
		self.generator = CovidTwitterPairwiseGenerator(self.config.hidden_size, self.config.hidden_dropout_prob)
		self.discriminator = CovidTwitterPairwiseDiscriminator(self.config.hidden_size, self.config.hidden_dropout_prob)
		self.d_rewards = None
		self.d_baseline = None
		self.g_loss = None
		self.d_ema = 0.99

	def training_step(self, batch, batch_idx, optimizer_idx):
		# [bsize], [bsize]
		outputs = self.bert(
			batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids']
		)
		contextualized_embeddings = outputs[0]

		logits, scores = self.generator(contextualized_embeddings)
		d_logits, d_probs = self.discriminator(contextualized_embeddings)

		# [bsize]
		labels = batch['labels']
		# [bsize]
		g_loss = self.generator.loss(logits, labels)
		# train generator
		if optimizer_idx == 0:
			self.g_loss = g_loss.detach()
			# d_scores is a probability distribution, therefore it acts as a weighted sum of losses.
			# [bsize] * [bsize] = [bsize]
			g_loss = g_loss * d_probs.detach()
			# [1]
			g_loss = g_loss.sum()

			self.log('train_loss', g_loss)
			return {
				'loss': g_loss
			}

		# train discriminator
		if optimizer_idx == 1:
			# d_range = np.arange(
			# 	start=0.00,
			# 	stop=1.00,
			# 	step=0.01
			# ).tolist()
			# d_max_metrics = self._get_max_metrics(scores, labels, name='train', threshold=d_range)
			# self.d_rewards = d_max_metrics['train_f1'].detach()
			g_loss = g_loss.detach()
			# difference in loss from single step
			g_loss_diff = g_loss - self.g_loss
			self.d_rewards = g_loss_diff

			if self.d_baseline is None:
				self.d_baseline = self.d_rewards.mean()
			# d_reward = (self.d_rewards - self.d_baseline)
			d_reward = self.d_rewards
			d_loss = self.discriminator.loss(d_probs, d_reward)
			d_max = torch.max(d_probs)
			d_loss = d_loss.mean()

			self.d_baseline = self.d_ema * self.d_baseline + (1.0 - self.d_ema) * self.d_rewards.mean()

			self.log('disc_loss', d_loss)
			self.log('disc_max_prob', d_max)
			self.log('disc_rewards', d_reward.mean())
			self.log('disc_baseline', self.d_baseline)
			# self.log('disc_threshold', d_max_metrics['train_threshold'])
			return {
				'loss': d_loss
			}

	def test_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'test')

	def validation_step(self, batch, batch_nb):
		return self._eval_step(batch, batch_nb, 'val')

	def _eval_step(self, batch, batch_nb, name):
		logits, scores = self(batch)

		if not self.predict_mode:
			labels = batch['labels']
			loss = self.generator.loss(logits, labels)
			loss = loss.detach()
			result = {
				f'{name}_loss': loss,
				f'{name}_batch_loss': loss,
				f'{name}_batch_scores': scores.detach(),
				f'{name}_batch_labels': batch['labels'].detach(),
			}

			return result
		else:
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

	def _get_max_metrics(self, scores, labels, name, threshold=None):
		if threshold is None:
			# cosine similarities between -1 and 1
			threshold_range = self._get_threshold_range()
		elif isinstance(threshold, list):
			threshold_range = threshold
		else:
			threshold_range = [threshold]
		max_metric = float('-inf')
		max_metrics = {}
		for threshold in threshold_range:
			t_metrics = self._get_metrics(scores, labels, threshold, name)
			m = t_metrics[f'{name}_f1']
			if m > max_metric:
				max_metric = m
				max_metrics = t_metrics
		return max_metrics

	def _eval_epoch_end(self, outputs, name):
		if not self.predict_mode:
			scores = torch.cat([x[f'{name}_batch_scores'].flatten() for x in outputs], dim=0)
			labels = torch.cat([x[f'{name}_batch_labels'].flatten() for x in outputs], dim=0)

			max_metrics = self._get_max_metrics(scores, labels, name, self.threshold)

			for metric, value in max_metrics.items():
				self.log(metric, value)

			loss = torch.cat([x[f'{name}_batch_loss'].flatten() for x in outputs], dim=0)
			loss = loss.mean()
			self.log(f'{name}_loss', loss)

	def validation_epoch_end(self, outputs):
		self._eval_epoch_end(outputs, 'val')

	def test_epoch_end(self, outputs):
		self._eval_epoch_end(outputs, 'test')

	def forward(self, batch):
		outputs = self.bert(
			batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids']
		)
		contextualized_embeddings = outputs[0]

		return self.generator(contextualized_embeddings)

	def _get_threshold_range(self):
		return np.arange(
			start=0.00,
			stop=1.00,
			step=0.001
		)

	def configure_optimizers(self):
		g_params = self._get_optimizer_params(self.weight_decay, modules=[self.bert, self.generator])
		g_optimizer = AdamW(
			g_params,
			lr=self.learning_rate,
			weight_decay=self.weight_decay,
			correct_bias=False
		)
		g_scheduler = get_linear_schedule_with_warmup(
			g_optimizer,
			num_warmup_steps=self.lr_warmup * self.updates_total,
			num_training_steps=self.updates_total
		)

		d_params = self._get_optimizer_params(self.weight_decay, modules=[self.bert, self.discriminator])
		d_optimizer = AdamW(
			d_params,
			lr=self.learning_rate,
			weight_decay=self.weight_decay,
			correct_bias=False
		)
		d_scheduler = get_linear_schedule_with_warmup(
			d_optimizer,
			num_warmup_steps=self.lr_warmup * self.updates_total,
			num_training_steps=self.updates_total
		)
		return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]

	def _get_optimizer_params(self, weight_decay, modules):
		param_optimizer = []
		for module in modules:
			param_optimizer.extend(list(module.named_parameters()))
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
