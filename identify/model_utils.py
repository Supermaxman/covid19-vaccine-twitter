
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
			self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total,
			weight_factor=1.0, threshold=None,
			torch_cache_dir=None, predict_mode=False, predict_path=None, load_pretrained=False
	):
		super().__init__()
		self.pre_model_name = pre_model_name
		self.torch_cache_dir = torch_cache_dir
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.lr_warmup = lr_warmup
		self.updates_total = updates_total
		self.weight_factor = weight_factor
		self.threshold = threshold
		self.predict_mode = predict_mode
		self.predict_path = predict_path
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
		self.criterion = nn.BCEWithLogitsLoss(reduction='none')
		self.score_func = torch.nn.Sigmoid()
		self.save_hyperparameters()

	def _forward_step(self, batch, batch_nb):
		ex_embs, m_embs, logits = self(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
			batch=batch
		)
		if not self.predict_mode:
			labels = batch['labels']
			loss = self._loss(
				logits,
				labels,
				self.weight_factor
			)
			# prediction = logits.max(dim=1)[1]
			prediction = (logits > 0).long()
			correct_count = ((labels.eq(1)).float() * (prediction.eq(labels)).float()).sum()
			total_count = (labels.eq(1)).float().sum()
			accuracy = correct_count / total_count
			if accuracy.isnan().item():
				accuracy = torch.zeros(1, dtype=torch.float)

			return loss, logits, prediction, correct_count, total_count, accuracy
		else:
			return logits, ex_embs, m_embs

	def training_step(self, batch, batch_nb):
		loss, logits, prediction, correct_count, total_count, accuracy = self._forward_step(batch, batch_nb)

		loss = loss.mean()
		self.log('train_loss', loss)
		self.log('train_accuracy', accuracy)
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
			loss, logits, prediction, correct_count, total_count, accuracy = self._forward_step(batch, batch_nb)

			result = {
				f'{name}_loss': loss.mean(),
				f'{name}_batch_loss': loss,
				f'{name}_batch_accuracy': accuracy,
				f'{name}_correct_count': correct_count,
				f'{name}_total_count': total_count,
				f'{name}_batch_logits': logits,
				f'{name}_batch_labels': batch['labels'],
				f'{name}_batch_predictions': prediction,
			}

			return result
		else:
			# TODO not correct for new formulation
			raise NotImplementedError()
			logits = self._forward_step(batch, batch_nb)
			logits = logits.detach()
			device_id = get_device_id()
			ex_dict = {
				'id': batch['id'],
				'question_id': batch['question_id'],
			}
			for i in range(logits.shape[-1]):
				ex_dict[f'{i}_score'] = logits[:, i].tolist()
			self.write_prediction_dict(
				ex_dict,
				filename=os.path.join(self.predict_path, f'predictions-{device_id}.pt')
			)
			result = {
				f'{name}_id': batch['id'],
				f'{name}_question_id': batch['question_id'],
				f'{name}_logits': logits,
			}

			return result

	def _get_predictions(self, logits, threshold):
		pos_probs = self.score_func(logits)
		predictions = (pos_probs.gt(threshold)).long()
		return predictions

	def _get_metrics(self, logits, labels, threshold, name):
		metrics = {}
		# [num_examples, num_misinfo]
		predictions = self._get_predictions(logits, threshold)
		# label is positive and predicted positive
		i_tp = (predictions.eq(1).float() * labels.eq(1).float()).sum()
		# label is not positive and predicted positive
		i_fp = (predictions.eq(1).float() * labels.ne(1).float()).sum()
		# label is positive and predicted negative
		i_fn = (predictions.ne(1).float() * labels.eq(1).float()).sum()
		i_precision = i_tp / (torch.clamp(i_tp + i_fp, 1.0))
		i_recall = i_tp / torch.clamp(i_tp + i_fn, 1.0)

		i_f1 = 2.0 * (i_precision * i_recall) / (torch.clamp(i_precision + i_recall, 1.0))
		macro_f1 = i_f1
		macro_p = i_precision
		macro_r = i_recall
		metrics[f'{name}_f1'] = i_f1
		metrics[f'{name}_p'] = i_precision
		metrics[f'{name}_r'] = i_recall

		metrics[f'{name}_macro_f1'] = macro_f1
		metrics[f'{name}_macro_p'] = macro_p
		metrics[f'{name}_macro_r'] = macro_r
		metrics[f'{name}_threshold'] = threshold
		return metrics

	def _eval_epoch_end(self, outputs, name):
		if not self.predict_mode:
			loss = torch.cat([x[f'{name}_batch_loss'].flatten() for x in outputs], dim=0).mean()
			logits = torch.cat([x[f'{name}_batch_logits'].flatten() for x in outputs], dim=0)
			labels = torch.cat([x[f'{name}_batch_labels'].flatten() for x in outputs], dim=0)

			if self.threshold is None:
				threshold_range = np.linspace(
					start=0.0,
					stop=1.0,
					num=100
				)
			else:
				threshold_range = [self.threshold]
			max_metric = float('-inf')
			max_metrics = {}
			for threshold in threshold_range:
				t_metrics = self._get_metrics(logits, labels, threshold, name)
				m = t_metrics[f'{name}_macro_f1']
				if m > max_metric:
					max_metric = m
					max_metrics = t_metrics

			for metric, value in max_metrics.items():
				self.log(metric, value)

			correct_count = torch.stack([x[f'{name}_correct_count'] for x in outputs], dim=0).sum()
			total_count = sum([x[f'{name}_total_count'] for x in outputs])
			accuracy = correct_count / total_count
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

	def _loss(self, logits, labels, weight_factor):
		loss = self.criterion(
			logits,
			labels
		)
		# if labels are 0 then weight is 1, otherwise weight is weight_factor
		l_weights = ((labels > 0).float() * weight_factor) + (labels.eq(0)).float()
		loss = loss * l_weights
		return loss


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
		# initialized value to 0.07
		self.temperature = Parameter(torch.ones(1, dtype=torch.float) * 0.07)

	def forward(self, input_ids, attention_mask, token_type_ids, batch):
		# [num_misinfo + bsize, seq_len, hidden_size]
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)
		contextualized_embeddings = outputs[0]
		# [num_misinfo + bsize, hidden_size]
		cls_output = contextualized_embeddings[:, 0]
		cls_output = self.f_dropout(cls_output)
		num_misinfo = batch['num_misinfo']

		# [bsize, hidden_size]
		ex_features = cls_output[num_misinfo:]
		# [bsize, emb_size]
		ex_embs = F.normalize(self.ex_embedding_layer(ex_features), p=2, dim=-1)

		# [num_misinfo, hidden_size]
		m_features = cls_output[:num_misinfo]
		# [num_misinfo, emb_size]
		m_embs = F.normalize(self.m_embedding_layer(m_features), p=2, dim=-1)
		# [bsize, emb_size] x [emb_size, num_misinfo] -> [bsize, num_misinfo]
		logits = torch.matmul(ex_embs, m_embs.t()) * torch.clamp(torch.exp(self.temperature), min=-100.0, max=100.0)
		return ex_embs, m_embs, logits


def get_device_id():
	try:
		device_id = dist.get_rank()
	except AssertionError:
		if 'XRT_SHARD_ORDINAL' in os.environ:
			device_id = int(os.environ['XRT_SHARD_ORDINAL'])
		else:
			device_id = 0
	return device_id
