
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
			print(f'predict_mode')
			# no need to load pre-trained weights since we will be loading whole model's
			# fine-tuned weights from checkpoint.
			self.config = BertConfig.from_pretrained(
				pre_model_name,
				cache_dir=torch_cache_dir
			)
			self.bert = BertModel(self.config)
		else:
			print(f'from_pretrained')
			self.bert = BertModel.from_pretrained(
				pre_model_name,
				cache_dir=torch_cache_dir
			)
			print(f'loaded')
			self.config = self.bert.config
		self.save_hyperparameters()
		self.batch_log = {}
		if 'binary_loss' in self.losses:
			self.bce_metric = torch.nn.BCEWithLogitsLoss(reduction='none')
			self.bias = Parameter(torch.zeros(1, dtype=torch.float))

	def _dim_loss(self, logits, labels_mask, dim):
		# non-positive logits are -1e9
		pos_logits = (logits + ((1.0 - labels_mask) * -1e9))
		# non-negative logits are -1e9
		neg_logits = (logits + (labels_mask * -1e9))
		eps = 1e-6
		# [ex_count, m_count]
		pos_loss = (-pos_logits) * labels_mask
		# [ex_count, m_count]
		norm_loss = torch.log(
			# [ex_count, 1] * [ex_count, m_count] -> [ex_count, m_count]
			torch.exp(neg_logits).sum(dim=dim, keepdim=True) + torch.exp(pos_logits) + eps
		) * labels_mask
		# [ex_count, m_count]
		loss = pos_loss + norm_loss

		return loss

	def _loss(self, logits, labels):
		print('_loss')
		loss = None
		labels_mask = labels.float()
		if 'compare_loss' in self.losses:
			# [1]
			m_pos_count = labels_mask.sum()

			# axis 1: each example over different misinfo
			# [ex_count, 1]
			m_loss = self._dim_loss(logits, labels_mask, dim=1)

			# axis 0: each misinfo over different examples
			# [1, m_count]
			ex_loss = self._dim_loss(logits, labels_mask, dim=0)

			c_loss = (m_loss + ex_loss) / 2
			c_loss = torch.sum(c_loss) / torch.clamp(m_pos_count, 1.0)

			if loss is None:
				loss = c_loss
			else:
				loss += c_loss

		if 'binary_loss' in self.losses:
			binary_logits = logits + self.bias
			bce_loss = self.bce_metric(
				binary_logits,
				labels_mask
			)
			bce_loss = torch.mean(bce_loss)
			if loss is None:
				loss = bce_loss
			else:
				loss += bce_loss

		return loss

	def _forward_step(self, batch, batch_nb):
		print('_forward_step')
		ex_embs, m_embs, logits, scores = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
			batch=batch
		)
		if not self.predict_mode:
			# [ex_count, m_count]
			labels = batch['labels']

			loss = self._loss(logits, labels)

			return loss, scores
		else:
			return ex_embs, m_embs, scores

	def training_step(self, batch, batch_nb):
		print('training_step')
		loss, scores = self._forward_step(batch, batch_nb)
		self.log('train_loss', loss)
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
		print('_eval_step')
		if not self.predict_mode:
			loss, scores = self._forward_step(batch, batch_nb)
			loss = loss.detach()
			result = {
				f'{name}_loss': loss,
				f'{name}_batch_loss': loss,
				f'{name}_batch_scores': scores.detach(),
				f'{name}_batch_labels': batch['labels'].detach(),
			}

			return result
		else:
			ex_embs, m_embs, scores = self._forward_step(batch, batch_nb)
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
			scores = torch.cat([x[f'{name}_batch_scores'].flatten() for x in outputs], dim=0)
			labels = torch.cat([x[f'{name}_batch_labels'].flatten() for x in outputs], dim=0)

			if self.threshold is None:
				# cosine similarities between -1 and 1
				threshold_range = self._get_threshold_range()
			else:
				threshold_range = [self.threshold]
			max_metric = float('-inf')
			max_metrics = {}
			for threshold in threshold_range:
				t_metrics = self._get_metrics(scores, labels, threshold, name)
				m = t_metrics[f'{name}_f1']
				if m > max_metric:
					max_metric = m
					max_metrics = t_metrics

			for metric, value in max_metrics.items():
				self.log(metric, value)

			loss = torch.cat([x[f'{name}_batch_loss'].flatten() for x in outputs], dim=0)
			loss = loss.mean()
			self.log(f'{name}_loss', loss)

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
			start=-1.00,
			stop=1.00,
			step=0.05
		)


class CovidTwitterMisinfoModel(BaseCovidTwitterMisinfoModel):
	def __init__(
			self, emb_size, *args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.emb_size = emb_size

		self.f_dropout = nn.Dropout(
			p=self.config.hidden_dropout_prob
		)
		self.ex_embedding_layer = nn.Linear(
			self.config.hidden_size,
			self.emb_size
		)
		self.m_embedding_layer = nn.Linear(
			self.config.hidden_size,
			self.emb_size
		)
		# initialized value to exp(x) = 0.07
		# self.temperature = Parameter(torch.log(torch.ones(1, dtype=torch.float) * 0.07))
		self.temperature = Parameter(torch.log(torch.ones(1, dtype=torch.float) / 0.07))
		self.batch_log['temperature'] = self.temperature

	def forward(self, input_ids, attention_mask, token_type_ids, batch):
		print(f'forward ({input_ids.shape}')
		# [num_misinfo + bsize, seq_len, hidden_size]
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		contextualized_embeddings = outputs[0]
		# [num_misinfo + bsize, hidden_size]
		lm_output = self._get_lm_output(contextualized_embeddings, attention_mask)
		lm_output = self.f_dropout(lm_output)
		num_misinfo = batch['num_misinfo']

		# [bsize, hidden_size]
		ex_features = lm_output[num_misinfo:]
		# [bsize, emb_size]
		ex_embs = F.normalize(self.ex_embedding_layer(ex_features), p=2, dim=-1)

		# [num_misinfo, hidden_size]
		m_features = lm_output[:num_misinfo]
		# [num_misinfo, emb_size]
		m_embs = F.normalize(self.m_embedding_layer(m_features), p=2, dim=-1)
		# -1 to 1
		scores = torch.matmul(ex_embs, m_embs.t())
		# [bsize, emb_size] x [emb_size, num_misinfo] -> [bsize, num_misinfo]
		logits = scores * torch.clamp(torch.exp(self.temperature), min=-100.0, max=100.0)
		# logits = scores / torch.exp(self.temperature)
		return ex_embs, m_embs, logits, scores

	def _get_lm_output(self, contextualized_embeddings, attention_mask):
		# cls embedding is first seq embedding
		# [b_size, seq_len, lm_size] -> [b_size, lm_size]
		cls_output = contextualized_embeddings[:, 0]
		return cls_output


class CovidTwitterStaticMisinfoModel(BaseCovidTwitterMisinfoModel):
	def __init__(
			self, num_misinfo, *args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.num_misinfo = num_misinfo

		self.f_dropout = nn.Dropout(
			p=self.config.hidden_dropout_prob
		)
		self.cls_layer = nn.Linear(
			self.config.hidden_size,
			self.num_misinfo,
			bias=False
		)
		self.bias = Parameter(torch.zeros(self.num_misinfo, dtype=torch.float))
		self.score_func = torch.nn.Sigmoid()

	def forward(self, input_ids, attention_mask, token_type_ids, batch):
		num_misinfo = batch['num_misinfo']
		# [bsize, seq_len, hidden_size]
		outputs = self.bert(
			input_ids[num_misinfo:],
			attention_mask=attention_mask[num_misinfo:],
			token_type_ids=token_type_ids[num_misinfo:]
		)
		contextualized_embeddings = outputs[0]
		# [bsize, hidden_size]
		lm_output = self._get_lm_output(contextualized_embeddings, attention_mask)
		lm_output = self.f_dropout(lm_output)
		# [bsize, num_misinfo]
		logits = self.cls_layer(lm_output)
		scores = self.score_func(logits + self.bias)

		return None, None, logits, scores

	def _get_lm_output(self, contextualized_embeddings, attention_mask):
		# cls embedding is first seq embedding
		# [b_size, seq_len, lm_size] -> [b_size, lm_size]
		cls_output = contextualized_embeddings[:, 0]
		return cls_output

	def _get_threshold_range(self):
		return np.arange(
			start=0.00,
			stop=1.00,
			step=0.0005
		)


class CovidTwitterMisinfoAvgModel(CovidTwitterMisinfoModel):
	def __init__(
			self, *args, **kwargs
	):
		super().__init__(*args, **kwargs)

	def _get_lm_output(self, contextualized_embeddings, attention_mask):
		# mean over sequence considering mask
		# [b_size, seq_len] -> [b_size, 1]
		seq_count = torch.sum(attention_mask.float(), dim=1, keepdim=True)
		# [b_size, seq_len, lm_size] -> [b_size, lm_size] / [b_size, 1] -> [b_size, lm_size]
		avg_output = torch.sum(contextualized_embeddings, dim=1) / seq_count
		return avg_output


class CovidTwitterPairwiseMisinfoModel(BaseCovidTwitterMisinfoModel):
	def __init__(
			self, *args, **kwargs
	):
		super().__init__(*args, **kwargs)

		self.f_dropout = nn.Dropout(
			p=self.config.hidden_dropout_prob
		)
		self.cls_classification_layer = nn.Linear(
			self.config.hidden_size,
			1,
			bias=False
		)

		self.score_func = torch.nn.Sigmoid()

	def forward(self, input_ids, attention_mask, token_type_ids, batch):
		# [num_misinfo + bsize, seq_len, hidden_size]
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		contextualized_embeddings = outputs[0]
		# [num_misinfo + bsize, hidden_size]
		lm_output = self._get_lm_output(contextualized_embeddings, attention_mask)
		lm_output = self.f_dropout(lm_output)
		# [bsize]
		logits = self.cls_classification_layer(lm_output).squeeze(dim=-1)
		scores = self.score_func(logits + self.bias)

		return None, None, logits, scores

	def _get_lm_output(self, contextualized_embeddings, attention_mask):
		# cls embedding is first seq embedding
		# [b_size, seq_len, lm_size] -> [b_size, lm_size]
		cls_output = contextualized_embeddings[:, 0]
		return cls_output

	def _get_threshold_range(self):
		return np.arange(
			start=0.00,
			stop=1.00,
			step=0.0005
		)


def get_device_id():
	try:
		device_id = dist.get_rank()
	except AssertionError:
		if 'XRT_SHARD_ORDINAL' in os.environ:
			device_id = int(os.environ['XRT_SHARD_ORDINAL'])
		else:
			device_id = 0
	return device_id
