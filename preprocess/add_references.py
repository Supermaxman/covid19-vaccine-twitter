
import argparse
import random

import numpy as np



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)
	parser.add_argument('-s', '--splits', default=5, type=float)
	parser.add_argument('-d', '--seed', default=0, type=float)
	parser.add_argument('-t', '--split_type', default='group')
	args = parser.parse_args()

	np.random.seed(args.seed)
	random.seed(args.seed)