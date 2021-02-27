
import os
import argparse
from filelock import FileLock
from datetime import datetime
import json
import base64


time_format = '%Y%m%d%H%M%S'

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-ex', '--experiment', required=True)
	parser.add_argument('-qp', '--queue_path', default='~/.default_queue')
	args = parser.parse_args()

	experiment = args.experiment
	queue_path = os.path.expanduser(args.queue_path)
	if not os.path.exists(queue_path):
		os.mkdir(queue_path)
	submitted_path = os.path.join(queue_path, 'submitted')
	if not os.path.exists(submitted_path):
		os.mkdir(submitted_path)
	ex_id = base64.urlsafe_b64encode(experiment.encode('utf-8')).decode('utf-8')
	status = {
		'status': 'submitted',
		'timestamp': datetime.now().strftime(time_format),
	}
	ex = {
		'ex_id': ex_id,
		'experiment': experiment,
		'process_id': None,
		'current_status': status,
		'status_history': [status]
	}
	if not os.path.exists(experiment):
		raise FileNotFoundError(f'Could not find experiment! {experiment}')

	ex_queue_path = os.path.join(submitted_path, ex_id)
	with FileLock(os.path.join(queue_path, '.lock')):
		if os.path.exists(ex_queue_path):
			print(f'Experiment already added to queue!')
		else:
			with open(ex_queue_path, 'w') as f:
				json.dump(ex, f, indent=4)
			print(f'Experiment successfully added to queue.')


