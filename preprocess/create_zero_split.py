
import os
import argparse
from sklearn.model_selection import train_test_split
import random
import numpy as np
import json


def read_jsonl(path):
	examples = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				ex = json.loads(line)
				examples.append(ex)
	return examples


def write_jsonl(data, path):
	with open(path, 'w') as f:
		for example in data:
			json_data = json.dumps(example)
			f.write(json_data + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-m', '--misinfo_path', required=True)
	parser.add_argument('-dm', '--dev_mids', required=True)
	parser.add_argument('-tm', '--test_mids', required=True)
	parser.add_argument('-s', '--seed', default=0, type=int)
	args = parser.parse_args()

	np.random.seed(args.seed)
	random.seed(args.seed)

	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)

	data = read_jsonl(args.input_path)

	with open(args.misinfo_path) as f:
		misinfo = json.load(f)

	dev_mids = set(args.dev_mids.split(','))
	test_mids = set(args.test_mids.split(','))

	train_misinfo = {m_id: m for m_id, m in misinfo.items() if m_id not in dev_mids and m_id not in test_mids}
	dev_misinfo = {m_id: m for m_id, m in misinfo.items() if m_id not in test_mids}
	test_misinfo = {m_id: m for m_id, m in misinfo.items()}

	train_data = []
	dev_data = []
	test_data = []

	for tweet in data:
		found_mid = False
		for m_id, m_label in tweet['misinfo'].items():
			if m_id in test_mids:
				test_data.append(tweet)
				found_mid = True
				break
			elif m_id in dev_mids:
				dev_data.append(tweet)
				found_mid = True
				break
		if not found_mid:
			train_data.append(tweet)

	print(f'Train size: {len(train_data)}, Dev size: {len(dev_data)}, Test size: {len(test_data)}')

	write_jsonl(train_data, os.path.join(args.output_path, 'train.jsonl'))
	with open(os.path.join(args.output_path, 'train_misinfo.json'), 'w') as f:
		json.dump(train_misinfo, f, indent=4)
	write_jsonl(dev_data, os.path.join(args.output_path, 'dev.jsonl'))
	with open(os.path.join(args.output_path, 'dev_misinfo.json'), 'w') as f:
		json.dump(dev_misinfo, f, indent=4)
	write_jsonl(test_data, os.path.join(args.output_path, 'test.jsonl'))
	with open(os.path.join(args.output_path, 'test_misinfo.json'), 'w') as f:
		json.dump(test_misinfo, f, indent=4)


