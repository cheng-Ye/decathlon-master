#!/usr/bin/env python3

import math

import torch

import pandas as pd
import models

from validate import validate
from util import  set_seed, get_trainable_params, count_params
from models.multitask_question_answering_network import MultitaskQuestionAnsweringNetwork
import os
#from multiprocess import Multiprocess, DistributedDataParallel

def to_iter(args, world_size, val_batch_size, data, train=True, token_testing=False, sort=None):
    sort = sort if not token_testing else True
    shuffle = None if not token_testing else False
    reverse = args.reverse
    Iterator = torchtext.data.BucketIterator if train else torchtext.data.Iterator
    it = Iterator(data, batch_size=val_batch_size, 
       device=0 if world_size > 0 else -1, batch_size_fn=batch_fn if train else None, 
       distributed=world_size>1, train=train, repeat=train, sort=sort, 
       shuffle=shuffle, reverse=args.reverse)
    return it

def get_learning_rate(i, warmup,dimension):
    return 0.1 * 10 / math.sqrt(dimension) * min(
        1 / math.sqrt(i), i / (warmup * math.sqrt(warmup)))

def step(model, batch, opt, iteration, lr=None, grad_clip=None,  it=None):
    model.train()
    opt.zero_grad()
    loss, predictions = model(batch)
    loss.backward()
    if lr is not None:
        opt.param_groups[0]['lr'] = lr
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm(model.params, grad_clip)
    opt.step()
    torch.save(model.state_dict(),'model.pth')
    return loss.data[0], {}

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]

def train(world_size=1):
    """main training function"""
    set_seed(123)              #设置随机数种子
    model = init_model(world_size)
    transformer_lr=True
    opt = init_opt(transformer_lr ,model) 
    iteration = 1

    

    getcwd=os.path.join(os.getcwd(),'csv')
    csvlist=os.listdir(getcwd)
    for task_id,task in enumerate(csvlist):

        task=current_csv=pd.read_csv('csv\\squad.csv',index_col=0,nrows=1000)
        
        data=chunks(task,50)                                           #batch_size=50

        for epoch in range(10):
            for i,batch in enumerate(data):
                #print(batch)
                #raise 1
                
                #val_loss, metric_dict = validate(val_task, val_iter, model, field, world_size, rank, num_print=args.num_print, args=args)
                #print ('val_loss',val_loss)
                
                # lr update
                lr = opt.param_groups[0]['lr'] 
                if  transformer_lr:
                    lr = get_learning_rate(iteration, 800,200) 

                # param update
                loss, train_metric_dict = step(model, batch, opt, iteration, lr=lr, grad_clip=1)

                print('task_id',task_id,'epoch:',epoch,'\nbatch',i,'loss:',loss)

def init_model( world_size):


    model =  MultitaskQuestionAnsweringNetwork()
    if os.path.isfile('model.pth'):
        print('load pretrained model')
        model.load_state_dict(torch.load('model.pth')) 
    else:
        print('new model ')
    params = get_trainable_params(model) 
    num_param = count_params(params)
    print(f'model  has {num_param:,} parameters')
    if world_size > 1: 
        print(f'Wrapping model for distributed')
        model = DistributedDataParallel(model)
    model.params = params
    return model


def init_opt(transformer_lr, model):
    opt = None
    if transformer_lr:
        opt = torch.optim.Adam(model.params, betas=(0.9, 0.98), eps=1e-9)
    else:
        opt = torch.optim.Adam(model.params, betas=(0.9, 0.999))
    return opt


if __name__ == '__main__':
    train()