# -*- coding: UTF-8 -*-
# !/usr/bin/env python3

import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.backbones import encoders
from copy import deepcopy
from collections import deque

class AveragedModel(nn.Module):
    def __init__(self, model, avg_fn=None, rm_optimizer=False):
        super(AveragedModel, self).__init__()
        model = self.filter_model(model)
        self.module = deepcopy(model)
        self.module.zero_grad(set_to_none=True)
        if rm_optimizer:
            for k, v in vars(self.module).items():
                if isinstance(v, torch.optim.Optimizer):
                    setattr(self.module, k, None)

        self.register_buffer("start_step", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("end_step", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("end_loss", torch.tensor(float('inf')))
        self.register_buffer("end_score", torch.tensor(0.0))
        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long))

        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (
                    num_averaged + 1
                )

        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @staticmethod
    def filter_model(model):

        if isinstance(model, AveragedModel):
            # prevent nested averagedmodel
            model = model.module

        if hasattr(model.module, "get_forward_model"):
            # default: model encapsulated by DataParallel
            model = model.module.get_forward_model()

        return model

    def update_parameters(self, model, step=None, start_step=None, end_step=None):
        """Update averaged model parameters

        Args:
            model: current model to update params
            step: current step. step is saved for log the averaged range
            start_step: set start_step only for first update
            end_step: set end_step
        """
        model = self.filter_model(model)
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(
                    self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device))
                )
        self.n_averaged += 1

        if step is not None:
            if start_step is None:
                start_step = step
            if end_step is None:
                end_step = step

        if start_step is not None:
            if self.n_averaged == 1:
                self.start_step.copy_(torch.tensor(start_step))

        if end_step is not None:
            self.end_step.copy_(torch.tensor(end_step))


def export_swad_model(avg_model, dataloader, train_swad, work_dir, logger):
    only_swad = train_swad.get('only_swad')
    start_iter = train_swad.get('start_iter', 0)
    logger.info('-'*50)
    logger.info(f'Export swad model...(only swad: {only_swad})')
    avg_model_path= os.path.join(work_dir, 'avg_model')
    assert os.path.exists(avg_model_path), 'The model was not trained with swad !'

    pth_list = []
    for pth in tqdm(os.listdir(avg_model_path)):
        name_info = re.findall(r'[-+]?\d+\.?\d*',pth)
        if len(name_info) >= 3:
            if int(name_info[0]) < start_iter:
                continue
            pth_data = [int(name_info[0]), float(name_info[1]), float(name_info[2]), pth]
        else:         
            pth_data = torch.load(os.path.join(avg_model_path, pth))
            if pth_data['iter'] < start_iter:
                continue
            pth_data = [pth_data['iter'], pth_data['state_dict'].get('end_loss'), pth_data['state_dict'].get('end_score'), pth]
            # pth_data = [pth_data['iter'], pth_data['val_loss'], pth_data['val_score'], pth]
        pth_list.append(pth_data)
    pth_list.sort(key=lambda x:x[0])
    logger.info('Iter\tVal_Loss\tVal_Score\tFilename')
    for pth_data in pth_list:
        logger.info(pth_data)
    
    def save_model(model, filename='swad_model.pth'):
        filename = os.path.join(work_dir, filename)
        checkpoint = dict(
            state_dict=model.module.state_dict())
        torch.save(checkpoint, filename)
        logger.info(f'save swad model: {filename}')

    mode = train_swad.get('mode', 'val_loss')
    if mode == 'val_loss':
        loss_list = np.array([row[1] for row in pth_list])
        min_loss = min(loss_list)
        threshold = min_loss * (1.0 + train_swad.get('tolerance_ratio', 0.2))
        min_arg = np.argmin(loss_list)
        logger.info(f'Iter {pth_list[min_arg][0]}, Min Loss: {min_loss}, Tolerance Threshold: {threshold}')

        swad_args = []
        for i in range(min_arg, -1, -1):
            if loss_list[i] > threshold:
                break   
            swad_args.append(i)
        n_tolerance = train_swad.get('n_tolerance', 16)
        if n_tolerance+min_arg > len(loss_list):
            logger.warning('The model may not have converged!')
        for i in range(min_arg+1, len(loss_list)):
            window = loss_list[i: i+n_tolerance]
            if min(window) > threshold:
                break
            swad_args.append(i)
        swad_args.sort()

        logger.info(f'Averaging the avg model with iter from {pth_list[swad_args[0]][0]} to {pth_list[swad_args[-1]][0]}, swad_thr:{threshold}')
        model_files = [pth_list[arg][3] for arg in swad_args]
        swad_model = AveragedModel(avg_model)
        for pth in tqdm(model_files):
            avg_model.load_state_dict(torch.load(os.path.join(avg_model_path, pth))['state_dict'], strict=False)
            swad_model.update_parameters(avg_model, start_step=avg_model.start_step.item(), end_step=avg_model.end_step.item())
        save_model(swad_model)
        if not train_swad.get('freeze_bn'):
            update_bn(dataloader, swad_model, train_swad.get('bn_training_epoch', 1))  ###
            save_model(swad_model, f'swad_bn_model.{time.time():.0f}.pth')


@torch.no_grad()
def update_bn(dataloader, swad, n_epochs):
    """
    Args:
        dataloader: train dataset dataloader
        swad: swad model
        n_epochs: epochs for BN statistics
    """
    model = swad.module
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        print('There is no BN.')
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for n in range(n_epochs):
        for data in tqdm(dataloader):
            data.pop('path', 'unknow')
            model(**data)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

