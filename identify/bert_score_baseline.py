
import os
import json
import argparse
import logging
import torch

from metric_utils import compute_threshold_f1
from data_utils import read_jsonl, label_text_to_relevant_id


def create_dataset(tweets, misinfo, tweet_scores):
	scores = torch.zeros([len(tweets), len(misinfo)], dtype=torch.float)
	labels = torch.zeros([len(tweets), len(misinfo)], dtype=torch.long)
	m_map = {m_id: m_idx for (m_idx, m_id) in enumerate(misinfo.keys())}
	for t_idx, t in enumerate(tweets):
		tweet_id = t['id']
		t_scores = tweet_scores[tweet_id]
		for m_id, m_label in t['misinfo'].items():
			m_label = label_text_to_relevant_id(m_label)
			m_score = t_scores[m_id]
			labels[t_idx, m_map[m_id]] = m_label
			scores[t_idx, m_map[m_id]] = m_score
	return labels, scores


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-tp', '--train_path', required=True)
	parser.add_argument('-vp', '--val_path', required=True)

	parser.add_argument('-sp', '--score_path', default='data/scores.json')
	parser.add_argument('-sd', '--save_directory', default='models')
	parser.add_argument('-mn', '--model_name', default='covid-twitter-v2-bertscore')
	parser.add_argument('-mip', '--misinfo_path', default=None)
	parser.add_argument('-th', '--threshold', default=None, type=float)

	args = parser.parse_args()

	save_directory = os.path.join(args.save_directory, args.model_name)
	checkpoint_path = os.path.join(save_directory, 'pytorch_model.bin')

	if not os.path.exists(save_directory):
		os.mkdir(save_directory)

	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)

	logfile = os.path.join(save_directory, "train_output.log")
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			logging.FileHandler(logfile, mode='w'),
			logging.StreamHandler()]
	)

	logging.info(f'Loading misinfo: {args.misinfo_path}')
	with open(args.misinfo_path, 'r') as f:
		misinfo = json.load(f)

	logging.info(f'Loading bertscore scores: {args.score_path}')
	with open(args.score_path, 'r') as f:
		scores = json.load(f)

	logging.info(f'Loading train dataset: {args.train_path}')
	train_data = read_jsonl(args.train_path)
	logging.info(f'Loading val dataset: {args.val_path}')
	val_data = read_jsonl(args.val_path)

	threshold = args.threshold
	if threshold is None:
		logging.info(f'Calculating training threshold...')
		t_labels, t_scores = create_dataset(train_data, misinfo, scores)
		_, _, _, threshold = compute_threshold_f1(
			scores=t_scores,
			labels=t_labels,
			threshold_min=-10.0,
			threshold_max=10.0,
			threshold_step=0.05
		)

	logging.info(f'Predicting on val data...')
	v_labels, v_scores = create_dataset(val_data, misinfo, scores)
	f1, p, r, _ = compute_threshold_f1(
		scores=v_scores,
		labels=v_labels,
		threshold=threshold
	)
	print(f'F1: {f1:.4f}, P: {p:.4f}, R: {r:.4f}, T: {threshold:.2f}')
