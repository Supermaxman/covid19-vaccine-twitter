
import os

import pytorch_lightning as pl
from transformers import BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist

import metric_utils
from emb_utils import *


class CovidTwitterMisinfoModel(pl.LightningModule):
	def __init__(
			self, pre_model_name, learning_rate, weight_decay, lr_warmup, updates_total, emb_model, emb_size, emb_loss_norm,
			gamma,
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
		self.emb_loss_norm = emb_loss_norm
		self.gamma = gamma

		self.f_dropout = nn.Dropout(
			p=self.config.hidden_dropout_prob
		)
		if emb_model == 'transd':
			self.emb_model = TransDEmbedding(
				self.config.hidden_size,
				self.emb_size,
				self.gamma,
				self.emb_loss_norm
			)
		elif emb_model == 'transe':
			self.emb_model = TransEEmbedding(
				self.config.hidden_size,
				self.emb_size,
				self.gamma,
				self.emb_loss_norm
			)
		elif emb_model == 'rotate':
			self.emb_model = RotatEEmbedding(
				self.config.hidden_size,
				self.emb_size,
				self.gamma,
				self.emb_loss_norm
			)
		elif emb_model == 'transms':
			self.emb_model = TransMSEmbedding(
				self.config.hidden_size,
				self.emb_size,
				self.gamma,
				self.emb_loss_norm
			)
		elif emb_model == 'tucker':
			self.emb_model = TuckEREmbedding(
				self.config.hidden_size,
				self.emb_size,
				self.gamma,
				self.emb_loss_norm
			)
		else:
			raise ValueError(f'Unknown embedding model: {emb_model}')

		self.save_hyperparameters()

	def forward(self, batch):
		num_examples = batch['num_examples']
		num_sequences_per_example = batch['num_sequences_per_example']
		num_entities = num_sequences_per_example - 1
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
		lm_output = lm_output.view(num_examples, num_sequences_per_example, self.config.hidden_size)
		# [bsize, hidden_size]
		r_lm_output = lm_output[:, 0]
		# [bsize * num_entities, hidden_size]
		e_lm_output = lm_output[:, 1:].reshape(num_examples * num_entities, self.config.hidden_size)
		e_embs = self.emb_model(e_lm_output, 'entity')
		# [bsize, num_entities, emb_size]
		e_embs = e_embs.view(num_examples, num_entities, self.emb_size)
		# [bsize, emb_size]
		r_embs = self.emb_model(r_lm_output, 'rel')
		# [bsize, num_seq, emb_size]
		return e_embs, r_embs

	def _triplet_energy(self, e_embs, m_embs, batch):
		# all [bsize, emb_size], [bsize, emb_size], [bsize, pos_samples, emb_size], [bsize, neg_samples, emb_size]
		t_ex_embs, pos_embs, neg_embs = self._split_embeddings(e_embs, batch)
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

	@staticmethod
	def _split_embeddings(embs, batch):
		t_ex_embs = embs[:, 0]
		pos_samples = batch['pos_samples']
		if pos_samples > 0:
			pos_embs = embs[:, 1:1+pos_samples]
		else:
			pos_embs = None
		neg_samples = batch['neg_samples']
		if neg_samples > 0:
			neg_embs = embs[:, 1+pos_samples:1+pos_samples+neg_samples]
		else:
			neg_embs = None
		return t_ex_embs, pos_embs, neg_embs

	def _loss(self, pos_energy, neg_energy, subj_obj_mask):
		# first randomly pick between subject and object losses
		# [bsize]
		pos_energy = (pos_energy * subj_obj_mask).sum(dim=-1)
		# [bsize]
		neg_energy = (neg_energy * subj_obj_mask).sum(dim=-1)

		loss, accuracy = self.emb_model.loss(pos_energy, neg_energy)

		loss = loss.mean()
		return loss, accuracy

	def _triplet_step(self, batch):
		e_embs, r_embs = self(batch)
		pos_energy, neg_energy = self._triplet_energy(e_embs, r_embs, batch)
		loss, accuracy = self._loss(pos_energy, neg_energy, batch['subj_obj_mask'])
		return loss, accuracy

	def training_step(self, batch, batch_nb):
		loss, accuracy = self._triplet_step(batch)
		self.log('train_loss', loss)
		self.log('train_accuracy', accuracy)
		result = {
			'loss': loss
		}
		return result

	def test_step(self, batch, batch_nb, dataloader_idx):
		return self._predict_step(batch, 'test')

	def validation_step(self, batch, batch_nb, dataloader_idx):
		if self.predict_mode:
			return self._predict_step(batch, 'val')
		else:
			if dataloader_idx == 0:
				return self._triplet_eval_step(batch, 'val')
			else:
				return self._predict_step(batch, 'val')

	def _predict_step(self, batch, name):
		e_type = batch['e_type']

		input_ids = batch['input_ids']
		attention_mask = batch['attention_mask']
		token_type_ids = batch['token_type_ids']

		contextualized_embeddings = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids
		)[0]
		# [bsize, hidden_size]
		lm_output = contextualized_embeddings[:, 0]
		b_embs = self.emb_model(lm_output, e_type)

		results = {
			f'{name}_e_type': e_type,
			f'{name}_ids': batch['ids'],
			f'{name}_b_embs': b_embs.detach(),
		}
		if 't_labels' in batch:
			results[f'{name}_t_labels'] = batch['t_labels']

		if 'm_examples' in batch:
			results[f'{name}_m_examples'] = batch['m_examples']
		return results

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

	def _val_epoch_end(self, outputs, name):
		triplet_eval_outputs, val_entity_outputs, val_rel_outputs = outputs
		# triplet eval is dataloader_idx 0
		loss = torch.cat([x[f'{name}_batch_loss'].flatten() for x in triplet_eval_outputs], dim=0)
		accuracy = torch.cat([x[f'{name}_batch_accuracy'].flatten() for x in triplet_eval_outputs], dim=0)
		loss = loss.mean()
		accuracy = accuracy.mean()
		self.log(f'{name}_loss', loss)
		self.log(f'{name}_accuracy', accuracy)

		dev_entities, dev_relations, dev_m_examples, dev_t_labels = self._extract_embeddings(
			val_entity_outputs,
			val_rel_outputs,
			name
		)
		# due to test val step causing missing entities at train start
		if len(dev_entities) < 100:
			return
		m_thresholds = metric_utils.find_m_thresholds(
			self.emb_model,
			dev_entities,
			dev_relations,
			dev_m_examples,
			dev_entities,
			dev_t_labels
		)

		f1, p, r, threshold = metric_utils.evaluate_m_thresholds(
			self.emb_model,
			dev_entities,
			dev_relations,
			dev_m_examples,
			dev_entities,
			dev_t_labels,
			m_thresholds
		)

		self.log(f'{name}_f1', f1)
		self.log(f'{name}_p', p)
		self.log(f'{name}_r', r)

	@staticmethod
	def _extract_embeddings(entity_outputs, rel_outputs, name):
		e_embs = torch.cat([x[f'{name}_b_embs'] for x in entity_outputs], dim=0)
		m_embs = torch.cat([x[f'{name}_b_embs'] for x in rel_outputs], dim=0)
		e_ids = [e_id for x in entity_outputs for e_id in x[f'{name}_ids']]
		m_ids = [m_id for x in rel_outputs for m_id in x[f'{name}_ids']]
		t_labels = [set(t_label.split(',')) for x in entity_outputs for t_label in x[f'{name}_t_labels']]
		m_examples = [m_ex.split(',') for x in rel_outputs for m_ex in x[f'{name}_m_examples']]
		entities = {e_id: e_emb for e_id, e_emb in zip(e_ids, e_embs)}
		relations = {r_id: r_emb for r_id, r_emb in zip(m_ids, m_embs)}
		t_labels = {t_id: t_l for t_id, t_l in zip(e_ids, t_labels)}
		m_examples = {m_id: m_e for m_id, m_e in zip(m_ids, m_examples)}
		return entities, relations, m_examples, t_labels

	def _test_epoch_end(self, outputs, name):
		val_entity_outputs, val_rel_outputs, test_entity_outputs, test_rel_outputs = outputs

		dev_entities, dev_relations, dev_m_examples, dev_t_labels = self._extract_embeddings(
			val_entity_outputs,
			val_rel_outputs,
			name
		)
		test_entities, test_relations, _, test_t_labels = self._extract_embeddings(
			test_entity_outputs,
			test_rel_outputs,
			name
		)

		m_thresholds = metric_utils.find_m_thresholds(
			self.emb_model,
			dev_entities,
			dev_relations,
			dev_m_examples,
			dev_entities,
			dev_t_labels
		)

		f1, p, r, threshold = metric_utils.evaluate_m_thresholds(
			self.emb_model,
			test_entities,
			test_relations,
			dev_m_examples,
			dev_entities,
			test_t_labels,
			m_thresholds
		)
		results = {
			'f1': f1,
			'p': p,
			'r': r,
			'm_thresholds': m_thresholds
		}

		self.log(f'{name}_f1', f1)
		self.log(f'{name}_p', p)
		self.log(f'{name}_r', r)
		return results

	def validation_epoch_end(self, outputs):
		if not self.predict_mode:
			self._val_epoch_end(outputs, 'val')
		else:
			self._predict_epoch_end(outputs, 'val')

	def test_epoch_end(self, outputs):
		if not self.predict_mode:
			return self._test_epoch_end(outputs, 'test')
		else:
			self._predict_epoch_end(outputs, 'test')

	def _predict_epoch_end(self, outputs, name):
		embs = torch.cat([x[f'{name}_b_embs'] for x in outputs], dim=0)
		e_ids = [e_id for x in outputs for e_id in x[f'{name}_ids']]
		e_type = outputs[0][f'{name}_e_type']

		device_id = get_device_id()
		e_path = os.path.join(self.predict_path, f'{e_type}-{device_id}-embeddings.pt')
		torch.save(
			{
				'ids': e_ids,
				'e_type': e_type,
				'embs': embs
			},
			e_path
		)

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
			num_warmup_steps=int(self.lr_warmup * self.updates_total),
			num_training_steps=self.updates_total
		)
		return [optimizer], [scheduler]

	def _get_optimizer_params(self, weight_decay):
		param_optimizer = list(self.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_params = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
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
