
from collections import defaultdict

import numpy as np
import torch


def get_predictions(logits, threshold):
	predictions = (logits.gt(threshold)).long()
	return predictions


def compute_f1(logits, labels, threshold):
	# [num_examples, num_misinfo]
	predictions = get_predictions(logits, threshold)
	# label is positive and predicted positive
	i_tp = (predictions.eq(1).float() * labels.eq(1).float()).sum()
	# label is not positive and predicted positive
	i_fp = (predictions.eq(1).float() * labels.ne(1).float()).sum()
	# label is positive and predicted negative
	i_fn = (predictions.ne(1).float() * labels.eq(1).float()).sum()
	i_precision = i_tp / (torch.clamp(i_tp + i_fp, 1.0))
	i_recall = i_tp / torch.clamp(i_tp + i_fn, 1.0)

	i_f1 = 2.0 * (i_precision * i_recall) / (torch.clamp(i_precision + i_recall, 1.0))
	return i_f1, i_precision, i_recall


def compute_threshold_f1(
		scores,
		labels,
		threshold=None,
		threshold_range=None,
		threshold_min=-1.0,
		threshold_max=1.0,
		threshold_step=0.05
):
	if threshold is None and threshold_range is None:
		threshold_range = np.arange(
			start=threshold_min,
			stop=threshold_max,
			step=threshold_step
		)
	elif threshold is not None:
		threshold_range = [threshold]
	max_f1 = float('-inf')
	max_vals = None
	for threshold in threshold_range:
		f1, p, r = compute_f1(scores, labels, threshold)
		if f1 > max_f1:
			max_f1 = f1
			max_vals = f1, p, r, threshold

	return max_vals


def find_m_thresholds(emb_model, entities, relations, m_examples, m_entities, t_labels):
	m_energies = defaultdict(list)
	m_labels = defaultdict(list)
	for t_id, t_emb in entities.items():
		for m_id, m_emb in relations.items():
			pos_t_ids = m_examples[m_id]
			m_label = 1 if m_id in t_labels[t_id] else 0
			p_e_embs = []
			for pos_t_id in pos_t_ids:
				p_e_emb = m_entities[pos_t_id]
				p_e_embs.append(p_e_emb)
			p_e_embs = torch.stack(p_e_embs, dim=0).mean(dim=0)
			p_e = emb_model.energy(
				head=t_emb,
				rel=m_emb,
				tail=p_e_embs
			)
			m_energies[m_id].append(p_e)
			m_labels[m_id].append(m_label)
	m_f_energies = {}
	m_f_labels = {}
	for m_id in m_energies:
		m_f_energies[m_id] = torch.tensor(m_energies[m_id], dtype=torch.float)
		m_f_labels[m_id] = torch.tensor(m_labels[m_id], dtype=torch.long)

	m_thresholds = {}
	for m_id, m_es in m_f_energies.items():
		scores = -m_es
		min_score = torch.min(scores).item()
		max_score = torch.max(scores).item()
		threshold_range = np.round(np.linspace(
			min_score,
			max_score,
			num=100
		), 4)
		f1, p, r, threshold = compute_threshold_f1(
			scores,
			m_f_labels[m_id],
			threshold_range=threshold_range,
		)

		m_thresholds[m_id] = threshold
	return m_thresholds


def evaluate_m_thresholds(emb_model, entities, relations, m_examples, m_entities, t_labels, m_thresholds):
	scores = []
	labels = []
	for t_id, t_emb in entities.items():
		for m_id, m_emb in relations.items():
			pos_t_ids = m_examples[m_id]
			m_label = 1 if m_id in t_labels[t_id] else 0
			p_e_embs = []
			for pos_t_id in pos_t_ids:
				p_e_emb = m_entities[pos_t_id]
				p_e_embs.append(p_e_emb)
			p_e_embs = torch.stack(p_e_embs, dim=0).mean(dim=0)
			p_e = emb_model.energy(
				head=t_emb,
				rel=m_emb,
				tail=p_e_embs
			)
			p_s = (-p_e).gt(m_thresholds[m_id]).float()
			scores.append(p_s)
			labels.append(m_label)
	scores = torch.tensor(scores, dtype=torch.float)
	labels = torch.tensor(labels, dtype=torch.long)

	min_score = torch.min(scores).item()
	max_score = torch.max(scores).item()
	threshold_range = np.round(np.linspace(
		min_score,
		max_score,
		num=100
	), 4)
	f1, p, r, threshold = compute_threshold_f1(
		scores,
		labels,
		threshold_range=threshold_range,
	)
	return f1, p, r, threshold
