
import os
import subprocess
import argparse
from filelock import FileLock


def reserve_gpus(request_count, gpu_mem_threshold, res_path):
	with FileLock(os.path.join(res_path, '.lock')):
		available_gpus = []
		for line in output:
			gpu_id, gpu_mem_info = line.split(',')
			gpu_mem = int(gpu_mem_info.split()[0])
			if gpu_mem < gpu_mem_threshold:
				gpu_res_path = os.path.join(res_path, gpu_id)
				if not os.path.exists(gpu_res_path):
					available_gpus.append(gpu_id)

		reserved_gpus = []
		if len(available_gpus) >= request_count:
			# reversed to allocate less used gpus on gpu cluster, avoid annoying others
			for gpu_id in list(reversed(available_gpus))[:request_count]:
				gpu_res_path = os.path.join(res_path, gpu_id)
				open(gpu_res_path, 'w').close()
				reserved_gpus.append(gpu_id)
	return reserved_gpus


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--request_count', required=True, type=int)
	parser.add_argument('-m', '--gpu_mem_threshold', default=200, type=int)
	parser.add_argument('-rp', '--res_path', default='~/.gpu_availability')
	args = parser.parse_args()

	command = 'nvidia-smi --query-gpu=index,memory.used --format=csv'
	process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	output = output.decode('utf-8').strip().split('\n')[1:]
	res_path = os.path.expanduser(args.res_path)
	if not os.path.exists(res_path):
		os.mkdir(res_path)
	if args.request_count == 0:
		print('', end='')
	else:
		gpu_ids = reserve_gpus(
			request_count=args.request_count,
			gpu_mem_threshold=args.gpu_mem_threshold,
			res_path=res_path,
		)

		if len(gpu_ids) > 0:
			print(','.join(gpu_ids), end='')
		else:
			print('-1', end='')

