# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *

class CrossEntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes, epsilon=0.1):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()
		return loss

class SoftEntropy(nn.Module):
	def __init__(self):
		super(SoftEntropy, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		loss = (- F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
		return loss

class NegEntropy(nn.Module):
	def __init__(self):
		super(NegEntropy, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs):
		log_probs = self.logsoftmax(inputs)
		loss = (F.softmax(inputs, dim=1) * log_probs).mean(0).sum()
		return loss

class FocalLoss(nn.Module):
	def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.eps = eps
		self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

	def forward(self, inputs, targets):
		log_probs = self.ce(inputs, targets)
		probs = torch.exp(-log_probs)
		loss = (1-probs)**self.gamma*log_probs
		return loss

class SoftFocalLoss(nn.Module):
	def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
		super(SoftFocalLoss, self).__init__()
		self.gamma = gamma
		self.eps = eps
		self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		# print("+++data+++")
		# print(inputs)
		log_probs = self.logsoftmax(inputs)
		# print("+++++++")
		# for log_prob in log_probs:
		# 	print(log_prob)
		loss = (- F.softmax(targets, dim=1).detach() * log_probs * ((1-F.softmax(inputs, dim=1))**self.gamma)).mean(0).sum()
		# print("loss")
		# print(loss)
		return loss
