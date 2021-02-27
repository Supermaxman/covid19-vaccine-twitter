
import os
import subprocess
import argparse
import time
import heapq
from datetime import datetime
import json

from filelock import FileLock


time_format = '%Y%m%d%H%M%S'


def update_status(queue_path, ex, status, p_id=None):
	old_file_path = os.path.join(queue_path, ex['current_status']['status'], ex['ex_id'])
	status_path = os.path.join(queue_path, status)
	if not os.path.exists(status_path):
		os.mkdir(status_path)
	file_path = os.path.join(status_path, ex['ex_id'])
	new_status = {
		'status': status,
		'timestamp': datetime.now().strftime(time_format),
	}
	ex['status_history'].insert(0, new_status)
	ex['current_status'] = new_status
	ex['process_id'] = p_id
	with open(file_path, 'w') as f:
		json.dump(ex, f, indent=4)
	os.remove(old_file_path)


def ex_format(ex):
	c_status = ex['current_status']
	p_id = ex['process_id']
	status = c_status['status']
	timestamp = datetime.strptime(c_status['timestamp'], time_format)
	experiment = ex['experiment']
	ex_id = ex['ex_id']
	return f'{timestamp} [{status}]{experiment} ({ex_id}) - {p_id}'


def get_experiments(queue_path, status):
	ex_list = []
	status_path = os.path.join(queue_path, status)
	if not os.path.exists(status_path):
		os.mkdir(status_path)
	for file in os.listdir(status_path):
		file_path = os.path.join(status_path, file)
		if os.path.isfile(file_path) and not file_path.endswith('.lock'):
			with open(file_path, 'r') as f:
				ex = json.load(f)
			c_status = ex['current_status']
			timestamp = datetime.strptime(c_status['timestamp'], time_format)
			ex_list.append((timestamp, ex))
	return ex_list


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-qp', '--queue_path', default='~/.default_queue')
	parser.add_argument('-mex', '--max_experiments', default=1, type=int)
	parser.add_argument('-rs', '--refresh_seconds', default=10, type=int)
	args = parser.parse_args()

	queue_path = os.path.expanduser(args.queue_path)

	if not os.path.exists(queue_path):
		os.mkdir(queue_path)

	max_experiments = args.max_experiments
	refresh_seconds = args.refresh_seconds
	processes = {p_id: None for p_id in range(max_experiments)}
	while True:
		with FileLock(os.path.join(queue_path, '.lock')):
			ex_queue = {
				'submitted': get_experiments(queue_path, 'submitted'),
				'running': get_experiments(queue_path, 'running'),
				# 'completed': []
			}

			num_available_processes = len(processes)
			for ts, ex in ex_queue['running']:
				p_id = ex['process_id']
				p = processes[p_id]
				if p is None:
					print(f'WARNING: running experiment lost process, resubmitting.')
					update_status(
						queue_path=queue_path,
						ex=ex,
						status='submitted',
						p_id=None
					)
				else:
					ret = p.poll()
					if ret is None:
						num_available_processes -= 1
					else:
						update_status(
							queue_path=queue_path,
							ex=ex,
							status='completed',
							p_id=None
						)
						processes[p_id] = None

			available_to_run_list = ex_queue['submitted']
			num_to_run = min(num_available_processes, len(available_to_run_list))
			if num_to_run > 0:
				top_to_run = heapq.nsmallest(num_to_run, available_to_run_list, key=lambda x: x[0])
				for ts, ex in top_to_run:
					p_id = min([p_id for p_id, p in processes.items() if p is None])
					experiment = ex['experiment']
					command = f'bash {experiment}'
					process = subprocess.Popen(command.split())
					processes[p_id] = process
					update_status(
						queue_path=queue_path,
						ex=ex,
						status='running',
						p_id=p_id
					)
		time.sleep(refresh_seconds)



