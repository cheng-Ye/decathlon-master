#!/usr/bin/env python3
import os
from text import torchtext
import torch
import numpy as np
import random

from util import get_splits, set_seed, preprocess_examples
from metrics import compute_metrics
import models
from models.multitask_question_answering_network import MultitaskQuestionAnsweringNetwork
from torch.nn import functional as F


def to_iter(data, bs):
    Iterator = torchtext.data.Iterator
    it = Iterator(data, batch_size=bs, 
       device=0, batch_size_fn=None, 
       train=False, repeat=False, sort=None, 
       shuffle=None, reverse=False)

    return it

if __name__ == '__main__':

    model =  MultitaskQuestionAnsweringNetwork()
    model.load_state_dict(torch.load('model.pth'))

    import pickle
    with open ('batch.pkl','rb') as f:
        batch=pickle.load(f)

    context, context_lengths, context_limited    = batch['context'],  batch['context_lengths'],  batch['context_limited']
    question, question_lengths, question_limited = batch['question'], batch['question_lengths'], batch['question_limited']
    #answer, answer_lengths, 
    answer_limited = batch['answer_limited']
    oov_to_limited_idx, limited_idx_to_full_idx  = batch['oov_to_limited_idx'], batch['limited_idx_to_full_idx ']
    
    #print( limited_idx_to_full_idx) 
    print(answer_limited[:10][:,1])
    model.eval()
    print('======================================')
    _, p = model(batch)
    print(p[:,1][:10])
    #loss = F.nll_loss(p[1][-2:-1,:].log(),answer_limited[:, 1:].contiguous().view(1,-1))



