
import json
import argparse


def write_jsonl(data, path):
	with open(path, 'w') as f:
		for example in data:
			json_data = json.dumps(example)
			f.write(json_data + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', required=True)
	parser.add_argument('-o', '--output_path', required=True)

	args = parser.parse_args()

	print('Loading misinfo...')
	with open(args.input_path, 'r') as f:
		misinfo = json.load(f)

	formatted_misinfo = []
	for m_id, m in misinfo.items():
		formatted_misinfo.append(
			{
				'id': m_id,
				'contents': m['text'],
			}
		)

	print('Writing jsonl misinfo...')
	write_jsonl(formatted_misinfo, args.output_path)

	print('Done!')
