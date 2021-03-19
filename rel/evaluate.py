
import os
import json
import argparse
import logging
import pytorch_lightning as pl
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers

from model_utils import *
from data_utils import *

import torch


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-vp', '--val_path', required=True)
	parser.add_argument('-tp', '--test_path', required=True)
	parser.add_argument('-pm', '--pre_model_name', default='nboost/pt-biobert-base-msmarco')
	parser.add_argument('-mn', '--model_name', default='pt-biobert-base-msmarco')
	parser.add_argument('-sd', '--save_directory', default='models')
	parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
	parser.add_argument('-ml', '--max_seq_len', default=96, type=int)
	parser.add_argument('-se', '--seed', default=0, type=int)
	parser.add_argument('-tpu', '--use_tpus', default=False, action='store_true')
	parser.add_argument('-gpu', '--gpus', default='0')
	parser.add_argument('-mip', '--misinfo_path', default=None)
	parser.add_argument('-es', '--emb_size', default=100, type=int)
	parser.add_argument('-eln', '--emb_loss_norm', default=2, type=int)
	parser.add_argument('-em', '--emb_model', default='transd')

	args = parser.parse_args()

	pl.seed_everything(args.seed)

	save_directory = os.path.join(args.save_directory, args.model_name)
	if not os.path.exists(save_directory):
		os.mkdir(save_directory)

	checkpoint_path = os.path.join(save_directory, 'pytorch_model.bin')
	results_path = os.path.join(save_directory, 'results.json')

	# export TPU_IP_ADDRESS=10.155.6.34
	# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
	gpus = [int(x) for x in args.gpus.split(',')]
	is_distributed = len(gpus) > 1
	precision = 16 if args.use_tpus else 32
	# precision = 32
	tpu_cores = 8
	num_workers = 4
	deterministic = True

	# Also add the stream handler so that it logs on STD out as well
	# Ref: https://stackoverflow.com/a/46098711/4535284
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)

	logfile = os.path.join(save_directory, "eval_output.log")
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			logging.FileHandler(logfile, mode='w'),
			logging.StreamHandler()]
	)

	logging.info(f'Loading tokenizer: {args.pre_model_name}')
	tokenizer = BertTokenizerFast.from_pretrained(args.pre_model_name)
	logging.info(f'Loading val dataset: {args.val_path}')
	val_data = read_jsonl(args.val_path)
	logging.info(f'Loading test dataset: {args.test_path}')
	test_data = read_jsonl(args.test_path)

	logging.info(f'Loading misinfo: {args.misinfo_path}')
	with open(args.misinfo_path, 'r') as f:
		misinfo = json.load(f)

		logging.info(f'Loaded misconception info.')

	logging.info('Loading datasets...')
	val_entity_dataset = MisinfoEntityDataset(
		documents=val_data,
		tokenizer=tokenizer,
		misinfo=misinfo
	)
	val_rel_dataset = MisinfoRelDataset(
		misinfo=misinfo,
		tokenizer=tokenizer,
		m_examples=val_entity_dataset.m_examples
	)
	val_entity_data_loader = DataLoader(
		val_entity_dataset,
		num_workers=num_workers,
		batch_size=args.eval_batch_size,
		shuffle=False,
		collate_fn=MisinfoPredictBatchCollator(
			args.max_seq_len,
			force_max_seq_len=args.use_tpus,
		)
	)
	val_rel_data_loader = DataLoader(
		val_rel_dataset,
		num_workers=num_workers,
		batch_size=args.eval_batch_size,
		shuffle=False,
		collate_fn=MisinfoPredictBatchCollator(
			args.max_seq_len,
			force_max_seq_len=args.use_tpus,
		)
	)
	test_entity_dataset = MisinfoEntityDataset(
		documents=test_data,
		tokenizer=tokenizer,
		misinfo=misinfo
	)
	test_rel_dataset = MisinfoRelDataset(
		misinfo=misinfo,
		tokenizer=tokenizer,
		m_examples=test_entity_dataset.m_examples
	)
	test_entity_data_loader = DataLoader(
		test_entity_dataset,
		num_workers=num_workers,
		batch_size=args.eval_batch_size,
		shuffle=False,
		collate_fn=MisinfoPredictBatchCollator(
			args.max_seq_len,
			force_max_seq_len=args.use_tpus,
		)
	)
	test_rel_data_loader = DataLoader(
		test_rel_dataset,
		num_workers=num_workers,
		batch_size=args.eval_batch_size,
		shuffle=False,
		collate_fn=MisinfoPredictBatchCollator(
			args.max_seq_len,
			force_max_seq_len=args.use_tpus,
		)
	)

	logging.info(f'val_entities={len(val_entity_dataset)}')
	logging.info(f'val_rels={len(val_rel_dataset)}')

	logging.info(f'test_entities={len(test_entity_dataset)}')
	logging.info(f'test_rels={len(test_rel_dataset)}')

	logging.info('Loading model...')
	model = CovidTwitterMisinfoModel(
		pre_model_name=args.pre_model_name,
		learning_rate=0,
		lr_warmup=0.0,
		updates_total=0,
		weight_decay=0,
		emb_model=args.emb_model,
		emb_size=args.emb_size,
		emb_loss_norm=args.emb_loss_norm,
		gamma=0.0,
		load_pretrained=True,
	)

	# load checkpoint
	logging.warning(f'Loading weights from trained checkpoint: {checkpoint_path}...')
	model.load_state_dict(torch.load(checkpoint_path))

	logger = pl_loggers.TensorBoardLogger(
		save_dir=save_directory,
		flush_secs=30,
		max_queue=2
	)

	if args.use_tpus:
		logging.warning('Gradient clipping slows down TPU training drastically, disabled for now.')
		trainer = pl.Trainer(
			logger=logger,
			tpu_cores=tpu_cores,
			default_root_dir=save_directory,
			max_epochs=0,
			precision=precision,
			deterministic=deterministic,
			checkpoint_callback=False,
		)
	else:
		if len(gpus) > 1:
			backend = 'ddp' if is_distributed else 'dp'
		else:
			backend = None
		trainer = pl.Trainer(
			logger=logger,
			gpus=gpus,
			default_root_dir=save_directory,
			max_epochs=0,
			precision=precision,
			distributed_backend=backend,
			deterministic=deterministic,
			checkpoint_callback=False,
		)

	logging.info('Evaluating...')
	try:
		results = trainer.test(
			model,
			test_dataloaders=[
				val_entity_data_loader,
				val_rel_data_loader,
				test_entity_data_loader,
				test_rel_data_loader,
			]
		)
		with open(results_path, 'w') as f:
			json.dump(results, f, indent=2)
		print(results[0])
		print(f'P\tR\tF1')
		print(f'{results[0]["test_p"]:.4f}\t{results[0]["test_r"]:.4f}\tt{results[0]["test_f1"]:.4f}')
	except Exception as e:
		logging.exception('Exception during evaluation', exc_info=e)


