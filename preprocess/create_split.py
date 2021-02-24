
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
	parser.add_argument('-ts', '--test_size', default=0.2, type=float)
	parser.add_argument('-ds', '--dev_size', default=0.1, type=float)
	parser.add_argument('-d', '--seed', default=0, type=int)
	args = parser.parse_args()

	np.random.seed(args.seed)
	random.seed(args.seed)

	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)

	data = read_jsonl(args.input_path)

	all_train_data, test_data = train_test_split(
		data,
		test_size=args.test_size
	)
	train_data, dev_data = train_test_split(
		all_train_data,
		test_size=args.dev_size
	)
	print(f'Train size: {len(train_data)}, Dev size: {len(dev_data)}, Test size: {len(test_data)}')

	write_jsonl(train_data, os.path.join(args.output_path, 'train.jsonl'))
	write_jsonl(dev_data, os.path.join(args.output_path, 'dev.jsonl'))
	write_jsonl(test_data, os.path.join(args.output_path, 'test.jsonl'))


