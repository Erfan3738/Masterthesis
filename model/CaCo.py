import torch
import torch.nn as nn
from torchvision.models import resnet
from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch.nn.functional as F

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)

class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, feature_dim=128, arch=None,bn_splits=16):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        norm_layer =  nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x

class CaCo(nn.Module):
   
    def __init__(self,args, dim=128, m=0.99, arch='resnet18',bn_splits=8):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(CaCo, self).__init__()
        self.args=args
        self.m = m
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = ModelBase(feature_dim=dim, arch=arch,bn_splits=bn_splits)
        #self.encoder_q.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.encoder_q.maxpool = nn.Identity()
        self.encoder_k = ModelBase(feature_dim=dim, arch=arch,bn_splits=bn_splits)
        #self.encoder_k.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.encoder_k.maxpool = nn.Identity()
        #dim_mlp = self.encoder_q.fc.weight.shape[1]
        # we do not keep 
        #self.encoder_q.fc = self._build_mlp(1,dim_mlp,args.mlp_dim,dim,last_bn=False)
        #self.encoder_k.fc = self._build_mlp(1, dim_mlp, args.mlp_dim, dim,last_bn=False)
        
        #self.encoder_q.fc = self._build_mlp(2,dim_mlp,args.mlp_dim,dim,last_bn=True)
        #self.encoder_k.fc = self._build_mlp(2, dim_mlp, args.mlp_dim, dim, last_bn=True)
        #self.predictor = self._build_mlp(2,dim,args.mlp_dim,dim,last_bn=False)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.K=args.cluster

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=True))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_param(self,moco_momentum):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * moco_momentum + param_q.data * (1. - moco_momentum)
           
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    

    def forward_withoutpred_sym(self,im_q,im_k,moco_momentum):
        q = self.encoder_q(im_q)#, use_feature=False)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        q_pred = q
        k_pred = self.encoder_q(im_k)#, use_feature=False)  # queries: NxC
        k_pred = nn.functional.normalize(k_pred, dim=1)
        with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
            self._momentum_update_key_encoder_param(moco_momentum)# update the key encoder
           
            im_q_, idx_unshuffle1 = self._batch_shuffle_single_gpu(im_q)
            q = self.encoder_k(im_q_)#, use_feature=False)  # keys: NxC
            q = nn.functional.normalize(q, dim=1)
            q = self._batch_unshuffle_single_gpu(q, idx_unshuffle1)
            q = q.detach()
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            k = self.encoder_k(im_k_)#, use_feature=False)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
            k = k.detach()

        return q_pred, k_pred, q, k
    def forward_withoutpred_multicrop(self,im_q_list,im_k,moco_momentum):
        q_list = []
        for im_q in [im_k]+im_q_list:
            q = self.encoder_q(im_q, use_feature=False)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            q_list.append(q)
        key_list = []
        with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
            self._momentum_update_key_encoder_param(moco_momentum)# update the key encoder
            for key_image in [im_k]+im_q_list[:1]:
                q = self.encoder_k(key_image, use_feature=False)  # keys: NxC
                q = nn.functional.normalize(q, dim=1)
                q = q.detach()
                key_list.append(q)
        return q_list,key_list
    

    def forward(self, im_q, im_k,run_type=0,moco_momentum=0.999):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            type: 0:sym; 1:multicrop
        Output:
            logits, targets

        """
        if run_type==0:
            return self.forward_withoutpred_sym(im_q,im_k,moco_momentum)
        elif run_type==1:
            return self.forward_withoutpred_multicrop(im_q, im_k, moco_momentum)



class CaCo_PN(nn.Module):
    def __init__(self,bank_size,dim):
        super(CaCo_PN, self).__init__()
        self.register_buffer("W", torch.randn(dim, bank_size))
        self.register_buffer("v", torch.zeros(dim, bank_size))
    def forward(self,q):
        memory_bank = self.W
        memory_bank = nn.functional.normalize(memory_bank, dim=0)
        logit=torch.einsum('nc,ck->nk', [q, memory_bank])
        return memory_bank, self.W, logit
    def update(self, m, lr, weight_decay, g):
        g = g + weight_decay * self.W
        self.v = m * self.v + g
        self.W = self.W - lr * self.v
    def print_weight(self):
        print(torch.sum(self.W).item())

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
