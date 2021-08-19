
import argparse
import json
import torch

from metric_utils import compute_f1


def read_jsonl(path):
	examples = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				examples.append(ex)
	return examples


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	# parser.add_argument('-o', '--output_path', required=True)
	args = parser.parse_args()
	all_preds = []
	for input_path in args.input_path.split(','):
		if input_path:
			print(f'{input_path}')
			with open(input_path) as f:
				# {'tweet_id': '1233992667366920192',
				#   'm_id': '127',
				#   'm_label': 0,
				#   'm_pred': 0},
				preds = read_jsonl(input_path)
				all_preds.extend(preds)

	labels = []
	model_preds = []
	m_ids = []
	t_ids = []
	for p in all_preds:
		labels.append(p['m_label'])
		model_preds.append(p['m_pred'])
		m_ids.append(p['m_id'])
		t_ids.append(p['tweet_id'])

	f1, p, r, _, _ = compute_f1(
		torch.tensor(model_preds),
		torch.tensor(labels),
		threshold=0
	)
	print(f'P\tR\tF1\tT')
	print(f'{p:.4f}\t{r:.4f}\t{f1:.4f}')
