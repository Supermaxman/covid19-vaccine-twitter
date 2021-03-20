
import math
from torch import nn
import torch
import numpy as np


class TransDEmbedding(nn.Module):
	def __init__(self, hidden_size, emb_size, gamma, loss_norm=2):
		super().__init__()
		self.gamma = gamma
		self.emb_size = emb_size
		self.loss_norm = loss_norm
		self.td_emb_size = self.emb_size // 2
		self.e_emb_layer = nn.Linear(
			hidden_size,
			self.td_emb_size
		)
		self.e_proj_layer = nn.Linear(
			hidden_size,
			self.td_emb_size
		)
		self.r_emb_layer = nn.Linear(
			hidden_size,
			self.td_emb_size
		)
		self.r_proj_layer = nn.Linear(
			hidden_size,
			self.td_emb_size
		)

	def forward(self, source_embeddings, emb_type):
		if emb_type == 'entity':
			# [bsize * num_seq, emb_size]
			ex_embs = self.e_emb_layer(source_embeddings)
			ex_projs = self.e_proj_layer(source_embeddings)
		elif emb_type == 'rel':
			# [bsize * num_seq, emb_size]
			ex_embs = self.r_emb_layer(source_embeddings)
			ex_projs = self.r_proj_layer(source_embeddings)
		else:
			raise ValueError(f'Unknown emb type: {emb_type}')
		# https://www.aclweb.org/anthology/P15-1067.pdf
		# normalize all lookups to max l2 norm of 1

		# [bsize * num_seq, emb_size]
		# ex_embs = ex_embs / torch.clamp(ex_emb_norms, min=1.0)
		ex_emb_norms = torch.norm(ex_embs, p=2, dim=-1, keepdim=True)
		ex_embs = ex_embs / ex_emb_norms

		ex_embs = torch.cat([ex_embs, ex_projs], dim=-1)
		return ex_embs

	@staticmethod
	def project(c, c_proj, r_proj):
		c_p = c + torch.sum(c * c_proj, dim=-1, keepdim=True) * r_proj
		c_p_norm = torch.norm(c_p, p=2, dim=-1, keepdim=True)
		# c_p = c_p / torch.clamp(c_p_norm, min=1.0)
		c_p = c_p / c_p_norm
		return c_p

	def energy(self, head, rel, tail):
		h, h_proj = head[..., :self.td_emb_size], head[..., self.td_emb_size:]
		r, r_proj = rel[..., :self.td_emb_size], rel[..., self.td_emb_size:]
		t, t_proj = tail[..., :self.td_emb_size], tail[..., self.td_emb_size:]
		h_p = self.project(h, h_proj, r_proj)
		t_p = self.project(t, t_proj, r_proj)
		h_r_t_diff = h_p + r - t_p
		# h_r_t_energy = torch.norm(h_r_t_diff, p=2, dim=-1, keepdim=False)
		# l2 norm squared = sum of squares
		if self.loss_norm == 1:
			h_r_t_energy = torch.norm(h_r_t_diff, p=1, dim=-1, keepdim=False)
		elif self.loss_norm == 2:
			h_r_t_energy = (h_r_t_diff * h_r_t_diff).sum(dim=-1)
		else:
			raise ValueError(f'Unknown loss norm: {self.loss_norm}')
		return h_r_t_energy

	def loss(self, pos_energy, neg_energy):
		margin = pos_energy - neg_energy
		loss = torch.clamp(self.gamma + margin, min=0.0)
		accuracy = (pos_energy.lt(neg_energy)).float().mean()
		return loss, accuracy


class TransEEmbedding(nn.Module):
	def __init__(self, hidden_size, emb_size, gamma, loss_norm=2):
		super().__init__()
		self.gamma = gamma
		self.emb_size = emb_size
		self.loss_norm = loss_norm
		self.e_emb_layer = nn.Linear(
			hidden_size,
			self.emb_size
		)
		self.r_emb_layer = nn.Linear(
			hidden_size,
			self.emb_size
		)

	def forward(self, source_embeddings, emb_type):
		if emb_type == 'entity':
			# [bsize * num_seq, emb_size]
			ex_embs = self.e_emb_layer(source_embeddings)
			ex_emb_norms = torch.norm(ex_embs, p=2, dim=-1, keepdim=True)
			# [bsize * num_seq, emb_size]
			ex_embs = ex_embs / ex_emb_norms
		elif emb_type == 'rel':
			# [bsize * num_seq, emb_size]
			ex_embs = self.r_emb_layer(source_embeddings)
		else:
			raise ValueError(f'Unknown emb type: {emb_type}')

		return ex_embs

	def energy(self, head, rel, tail):
		h_r_t_diff = head + rel - tail
		if self.loss_norm == 1:
			h_r_t_energy = torch.norm(h_r_t_diff, p=1, dim=-1, keepdim=False)
		elif self.loss_norm == 2:
			h_r_t_energy = (h_r_t_diff * h_r_t_diff).sum(dim=-1)
		else:
			raise ValueError(f'Unknown loss norm: {self.loss_norm}')
		return h_r_t_energy

	def loss(self, pos_energy, neg_energy):
		margin = pos_energy - neg_energy
		loss = torch.clamp(self.gamma + margin, min=0.0)
		accuracy = (pos_energy.lt(neg_energy)).float().mean()
		return loss, accuracy


class RotatEEmbedding(nn.Module):
	def __init__(self, hidden_size, emb_size, gamma, loss_norm=2):
		super().__init__()
		self.gamma = gamma
		self.emb_size = emb_size
		self.loss_norm = loss_norm
		self.td_emb_size = self.emb_size // 2
		self.e_emb_layer = nn.Linear(
			hidden_size,
			self.td_emb_size
		)
		self.e_proj_layer = nn.Linear(
			hidden_size,
			self.td_emb_size
		)
		self.r_emb_layer = nn.Linear(
			hidden_size,
			self.td_emb_size
		)

	def forward(self, source_embeddings, emb_type):
		if emb_type == 'entity':
			# [bsize * num_seq, emb_size]
			ex_embs = self.e_emb_layer(source_embeddings)
			ex_ims = self.e_proj_layer(source_embeddings)
			ex_embs = torch.cat([ex_embs, ex_ims], dim=-1)
			return ex_embs
		elif emb_type == 'rel':
			# [bsize * num_seq, emb_size]
			r_embs = self.r_emb_layer(source_embeddings)
			return r_embs
		else:
			raise ValueError(f'Unknown emb type: {emb_type}')

	def energy(self, head, rel, tail):
		h_re, h_im = head[..., :self.td_emb_size], head[..., self.td_emb_size:]
		t_re, t_im = tail[..., :self.td_emb_size], tail[..., self.td_emb_size:]
		r_phase = torch.tanh(rel) * math.pi
		r_re = torch.cos(r_phase)
		r_im = torch.sin(r_phase)

		re_score = (h_re * r_re - h_im * r_im) - t_re
		im_score = (h_re * r_im + h_im * r_re) - t_im
		h_r_t_diff = torch.cat([re_score, im_score], dim=-1)

		# l2 norm squared = sum of squares
		if self.loss_norm == 1:
			h_r_t_energy = torch.norm(h_r_t_diff, p=1, dim=-1, keepdim=False)
		elif self.loss_norm == 2:
			h_r_t_energy = (h_r_t_diff * h_r_t_diff).sum(dim=-1)
		else:
			raise ValueError(f'Unknown loss norm: {self.loss_norm}')
		return h_r_t_energy

	def loss(self, pos_energy, neg_energy):
		# # pos_loss = -torch.log(torch.sigmoid(self.gamma - pos_energy) + 1e-6)
		# # neg_loss = -torch.log(torch.sigmoid(neg_energy - self.gamma) + 1e-6)
		# # loss = pos_loss + neg_loss
		# accuracy = (pos_energy.lt(neg_energy)).float().mean()
		margin = pos_energy - neg_energy
		loss = torch.clamp(self.gamma + margin, min=0.0)
		accuracy = (pos_energy.lt(neg_energy)).float().mean()
		return loss, accuracy


class TransMSEmbedding(nn.Module):
	def __init__(self, hidden_size, emb_size, gamma, loss_norm=2):
		super().__init__()
		self.gamma = gamma
		self.emb_size = emb_size
		self.loss_norm = loss_norm
		self.e_emb_layer = nn.Linear(
			hidden_size,
			self.emb_size
		)
		self.r_emb_layer = nn.Linear(
			hidden_size,
			self.emb_size + 1
		)

	def forward(self, source_embeddings, emb_type):
		if emb_type == 'entity':
			# [bsize * num_seq, emb_size]
			ex_embs = self.e_emb_layer(source_embeddings)
			ex_emb_norms = torch.norm(ex_embs, p=2, dim=-1, keepdim=True)
			# [bsize * num_seq, emb_size]
			ex_embs = ex_embs / ex_emb_norms
		elif emb_type == 'rel':
			# [bsize * num_seq, emb_size]
			ex_embs = self.r_emb_layer(source_embeddings)
		else:
			raise ValueError(f'Unknown emb type: {emb_type}')

		return ex_embs

	def energy(self, head, rel, tail):
		rel = rel[..., :self.emb_size]
		alpha = rel[..., -1]
		alpha = alpha.unsqueeze(dim=-1)
		h_p = -torch.tanh(tail * rel) * head
		r_p = rel + alpha * (head * tail)
		t_p = torch.tanh(head * rel) * tail

		h_r_t_diff = h_p + r_p - t_p
		if self.loss_norm == 1:
			h_r_t_energy = torch.norm(h_r_t_diff, p=1, dim=-1, keepdim=False)
		elif self.loss_norm == 2:
			h_r_t_energy = (h_r_t_diff * h_r_t_diff).sum(dim=-1)
		else:
			raise ValueError(f'Unknown loss norm: {self.loss_norm}')
		return h_r_t_energy

	def loss(self, pos_energy, neg_energy):
		margin = pos_energy - neg_energy
		loss = torch.clamp(self.gamma + margin, min=0.0)
		accuracy = (pos_energy.lt(neg_energy)).float().mean()
		return loss, accuracy


class TuckEREmbedding(nn.Module):
	def __init__(self, hidden_size, emb_size, gamma, loss_norm=2):
		super().__init__()
		self.gamma = gamma
		self.emb_size = emb_size
		self.loss_norm = loss_norm

		self.weight = nn.parameter.Parameter(
			torch.tensor(np.random.uniform(-1, 1, (self.emb_size, self.emb_size, self.emb_size)))
		)

		self.e_emb_layer = nn.Linear(
			hidden_size,
			self.emb_size
		)
		self.r_emb_layer = nn.Linear(
			hidden_size,
			self.emb_size
		)
		# self.score_func = nn.LogSigmoid()

	def forward(self, source_embeddings, emb_type):
		if emb_type == 'entity':
			# [bsize * num_seq, emb_size]
			ex_embs = self.e_emb_layer(source_embeddings)
			return ex_embs
		elif emb_type == 'rel':
			# [bsize * num_seq, emb_size]
			r_embs = self.r_emb_layer(source_embeddings)
			return r_embs
		else:
			raise ValueError(f'Unknown emb type: {emb_type}')

	def energy(self, head, rel, tail):
		num_batch_dims = len(head.shape) - 1
		w = self.weight
		# [1, 1, ..., e_size, e_size, e_size]
		for _ in range(num_batch_dims):
			w = w.unsqueeze(dim=0)
		head = head.unsqueeze(dim=-1)
		head = head.unsqueeze(dim=-1)
		# [..., emb_size, emb_size, emb_size] -> [..., emb_size, emb_size]
		w = (w * head).sum(dim=-1)
		# [..., emb_size, 1]
		rel = rel.unsqueeze(dim=-1)
		# [..., emb_size]
		w = (w * rel).sum(dim=-1)
		# [...]
		w = (w * tail).sum(dim=-1)
		# this is treated as a score by TuckER, so - makes energy
		h_r_t_energy = -w
		return h_r_t_energy

	def loss(self, pos_energy, neg_energy):

		# pos_loss = -torch.log(torch.sigmoid(-pos_energy) + 1e-6)
		# neg_loss = -torch.log(1.0 - torch.sigmoid(-neg_energy) + 1e-6)
		# loss = pos_loss + neg_loss
		# accuracy = (pos_energy.lt(neg_energy)).float().mean()
		# return loss, accuracy

		margin = pos_energy - neg_energy
		loss = torch.clamp(self.gamma + margin, min=0.0)
		accuracy = (pos_energy.lt(neg_energy)).float().mean()
		return loss, accuracy
