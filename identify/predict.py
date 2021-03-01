
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
	parser.add_argument('-op', '--output_path', required=True)
	parser.add_argument('-pm', '--pre_model_name', default='nboost/pt-biobert-base-msmarco')
	parser.add_argument('-mn', '--model_name', default='pt-biobert-base-msmarco')
	parser.add_argument('-sd', '--save_directory', default='models')
	parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
	parser.add_argument('-ml', '--max_seq_len', default=96, type=int)
	parser.add_argument('-se', '--seed', default=0, type=int)
	parser.add_argument('-cd', '--torch_cache_dir', default=None)
	parser.add_argument('-tpu', '--use_tpus', default=False, action='store_true')
	parser.add_argument('-gpu', '--gpus', default='0')
	parser.add_argument('-ts', '--train_sampling', default='none')
	parser.add_argument('-ls', '--losses', default='compare_loss')
	parser.add_argument('-mip', '--misinfo_path', default=None)
	parser.add_argument('-mt', '--model_type', default='lm')
	parser.add_argument('-es', '--emb_size', default=100, type=int)

	args = parser.parse_args()

	pl.seed_everything(args.seed)

	save_directory = os.path.join(args.save_directory, args.model_name)
	checkpoint_path = os.path.join(save_directory, 'pytorch_model.bin')

	if not os.path.exists(save_directory):
		os.mkdir(save_directory)

	checkpoint_path = os.path.join(save_directory, 'pytorch_model.bin')

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

	logfile = os.path.join(save_directory, "train_output.log")
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

	logging.info(f'Loading misinfo: {args.misinfo_path}')
	with open(args.misinfo_path, 'r') as f:
		misinfo = json.load(f)

		logging.info(f'Loaded misconception info.')
	logging.info('Loading datasets...')
	train_sampling = args.train_sampling.lower()
	if train_sampling == 'pairwise':
		val_dataset = MisinfoPairwiseDataset(
			documents=val_data,
			tokenizer=tokenizer,
			misinfo=misinfo,
			all_misinfo=True
		)
		val_data_loader = DataLoader(
			val_dataset,
			num_workers=num_workers,
			shuffle=False,
			batch_size=args.eval_batch_size,
			collate_fn=MisinfoPairwiseBatchCollator(
				misinfo,
				tokenizer,
				args.max_seq_len,
				all_misinfo=True,
				force_max_seq_len=args.use_tpus,
			)
		)
	elif train_sampling == 'pairwise-emb':
		val_dataset = MisinfoPairwiseEmbDataset(
			documents=val_data,
			tokenizer=tokenizer,
			misinfo=misinfo,
			all_misinfo=True
		)
		val_data_loader = DataLoader(
			val_dataset,
			num_workers=num_workers,
			shuffle=False,
			batch_size=args.eval_batch_size,
			collate_fn=MisinfoPairwiseEmbBatchCollator(
				misinfo,
				tokenizer,
				args.max_seq_len,
				all_misinfo=True,
				force_max_seq_len=args.use_tpus,
			)
		)
	else:
		val_dataset = MisinfoDataset(
			documents=val_data,
			tokenizer=tokenizer,
			misinfo=misinfo
		)
		val_data_loader = DataLoader(
			val_dataset,
			num_workers=num_workers,
			shuffle=False,
			batch_size=args.eval_batch_size,
			collate_fn=MisinfoBatchCollator(
				misinfo,
				tokenizer,
				args.max_seq_len,
				all_misinfo=True,
				force_max_seq_len=args.use_tpus,
			)
		)

	logging.info(f'train_sampling={train_sampling}')
	logging.info(f'val={len(val_dataset)}')

	logging.info('Loading model...')
	model_args = dict(
		pre_model_name=args.pre_model_name,
		learning_rate=0,
		lr_warmup=0.1,
		updates_total=0,
		weight_decay=0,
		losses=args.losses.split(','),
		torch_cache_dir=args.torch_cache_dir,
		load_pretrained=True,
		predict_mode=True,
		predict_path=args.output_path
	)

	model_type = args.model_type.lower()
	if model_type == 'lm':
		model = CovidTwitterMisinfoModel(
			**model_args,
			emb_size=args.emb_size
		)
	elif model_type == 'lm-avg':
		model = CovidTwitterMisinfoAvgModel(
			**model_args,
			emb_size=args.emb_size
		)
	elif model_type == 'lm-pairwise':
		model = CovidTwitterPairwiseMisinfoModel(
			**model_args
		)
	elif model_type == 'lm-pairwise-emb':
		model = CovidTwitterPairwiseEmbMisinfoModel(
			**model_args,
			emb_size=args.emb_size
		)
	elif model_type == 'lm-static':
		model = CovidTwitterStaticMisinfoModel(
			**model_args,
			num_misinfo=len(misinfo)
		)
	else:
		raise ValueError(f'Unknown model type: {model_type}')

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

	logging.info('Predicting...')
	try:
		trainer.test(model, val_data_loader)
	except Exception as e:
		logging.exception('Exception during predicting', exc_info=e)


