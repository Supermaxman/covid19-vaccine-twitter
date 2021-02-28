import torch
import argparse
from collections import defaultdict
import os
import json


def load_predictions(input_path):
	pred_list = []
	for file_name in os.listdir(input_path):
		if file_name.endswith('.pt'):
			preds = torch.load(os.path.join(input_path, file_name))
			pred_list.extend(preds)
	m_scores = defaultdict(dict)
	for prediction in pred_list:
		t_id = prediction['id']
		m_id = prediction['m_id']
		m_score = prediction['m_score']

		m_scores[t_id][m_id] = m_score

	return m_scores


def save_predictions(question_scores, output_path, run_name):
	with open(output_path, 'w') as f:
		for question_id, question_scores in question_scores.items():
			for idx, (doc_pass_id, score) in enumerate(question_scores):
				rank = idx + 1
				f.write(f'{question_id}\tQ0\t{doc_pass_id}\t{rank}\t{score:.8f}\t{run_name}\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()

	input_path = args.input_path
	output_path = args.output_path
	output_folder = os.path.dirname(output_path)
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
	m_scores = load_predictions(input_path)
	with open(output_path, 'w') as f:
		json.dump(m_scores, f)
