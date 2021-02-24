
import os
import argparse
from filelock import FileLock


def free_gpus(gpu_ids, res_path):
	with FileLock(os.path.join(res_path, '.lock')):
		for gpu_id in gpu_ids:
			gpu_res_path = os.path.join(res_path, gpu_id)
			if os.path.exists(gpu_res_path):
				os.remove(gpu_res_path)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--gpu_ids', required=True)
	parser.add_argument('-rp', '--res_path', default='~/.gpu_availability')
	args = parser.parse_args()

	gpu_ids = args.gpu_ids.split(',')
	full_res_path = os.path.expanduser(args.res_path)
	free_gpus(gpu_ids, full_res_path)


