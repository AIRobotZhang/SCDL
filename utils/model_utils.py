# -*- coding:utf-8 -*-
import logging
import os
import json
import torch.nn.functional as F
import torch
import numpy as np

logger = logging.getLogger(__name__)

def soft_frequency(logits, power=2, probs=False):
    """
    Unsupervised Deep Embedding for Clustering Analysiszaodian
    https://arxiv.org/abs/1511.06335
    """
    if not probs:
        softmax = torch.nn.Softmax(dim=1)
        y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
    else:
        y = logits
    f = torch.sum(y, dim=(0, 1))
    t = y**power / f
    p = t/torch.sum(t, dim=2, keepdim=True)
    # m = torch.argmax(y, dim=2, keepdim=True)
    # m = (m==0)
    # m = m.repeat(1,1,y.size(2))
    # p = p.masked_fill(mask=m,value=torch.tensor(0))
    # m = ~m
    # y = y.masked_fill(mask=m,value=torch.tensor(0))
    # p = p+y

    return p

def get_hard_label(args, combined_labels, pred_labels, pad_token_label_id, pred_logits=None):
    pred_labels[combined_labels==pad_token_label_id] = pad_token_label_id

    return pred_labels, None

def mask_tokens(args, combined_labels, pred_labels, pad_token_label_id, pred_logits=None):

    if args.self_learning_label_mode == "hard":
        softmax = torch.nn.Softmax(dim=1)
        y = softmax(pred_logits.view(-1, pred_logits.shape[-1])).view(pred_logits.shape)
        _threshold = args.threshold
        pred_labels[y.max(dim=-1)[0]>_threshold] = pad_token_label_id
        # if args.self_training_hp_label < 5:
        #     pred_labels[combined_labels==pad_token_label_id] = pad_token_label_id
        # pred_labels[combined_labels==pad_token_label_id] = pad_token_label_id
        return pred_labels, None

    elif args.self_learning_label_mode == "soft":
        label_mask = (pred_labels.max(dim=-1)[0]>args.threshold)
        
        return pred_labels, label_mask

def opt_grad(loss, in_var, optimizer):
    
    if hasattr(optimizer, 'scalar'):
        loss = loss * optimizer.scaler.loss_scale
    return torch.autograd.grad(loss, in_var)

def _update_mean_model_variables(model, m_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for m_param, param in zip(m_model.parameters(), model.parameters()):
        m_param.data.mul_(alpha).add_(1 - alpha, param.data)

def _update_mean_prediction_variables(prediction, m_prediction, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # for m_param, param in zip(m_model.parameters(), model.parameters()):
    m_prediction.data.mul_(alpha).add_(1 - alpha, prediction.data)
