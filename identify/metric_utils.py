
import numpy as np
import torch


def get_predictions(logits, threshold):
	# normalize over
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
	return i_f1, i_precision, i_recall, threshold


def compute_threshold_f1(
		scores,
		labels,
		threshold=None,
		threshold_min=-1.0,
		threshold_max=1.0,
		threshold_step=0.05
):
	if threshold is None:
		# cosine similarities between -1 and 1
		threshold_range = np.arange(
			start=threshold_min,
			stop=threshold_max,
			step=threshold_step
		)
	else:
		threshold_range = [threshold]
	max_f1 = float('-inf')
	max_vals = None
	for threshold in threshold_range:
		f1, p, r, t = compute_f1(scores, labels, threshold)
		if f1 > max_f1:
			max_f1 = f1
			max_vals = f1, p, r, t

	return max_vals
