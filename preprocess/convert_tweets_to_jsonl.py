
import json
import argparse


def read_jsonl(path):
	examples = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				try:
					ex = json.loads(line)
					examples.append(ex)
				except Exception as e:
					print(e)
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

	args = parser.parse_args()

	print('Loading tweets...')
	tweets = read_jsonl(args.input_path)

	formatted_tweets = []
	for tweet in tweets:
		formatted_tweets.append(
			{
				'id': tweet['id'],
				'contents': tweet['full_text'],
			}
		)

	print('Writing jsonl tweets...')
	write_jsonl(formatted_tweets, args.output_path)

	print('Done!')
