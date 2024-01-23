from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import numpy as np
import math
import torch
import torch.nn as nn
import random
from torch.multiprocessing import cpu_count
from torch.optim import Adam
import pytorch_lightning as pl
from argparse import Namespace
import argparse
import random
import pickle

from architect import *
from ios import read_data_new, read_coverage, read_fragment, combine_reps

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha)

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class ContrastLearn(pl.LightningModule):
    def __init__(self, hparams):
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        super().__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.modelname == 'ResAE':
            self.model = ResAE(out_channels=self.hparams.hidden_size, embedding_size=self.hparams.emb_size, 
                               cluster_num=self.hparams.class_num, 
                               kernel_size=self.hparams.first_kernel_size, 
                               input_size = self.hparams.input_size)
        if self.hparams.modelname == 'CNNAE':
            emb = DeepAutoencoder(self.hparams.hidden_size, self.hparams.emb_size, 
                                  self.hparams.first_kernel_size, self.hparams.input_size)
            self.model = Network2(emb, self.hparams.fea_dim, self.hparams.class_num)
            
        self.ins_loss = InstanceLoss(self.hparams.ins_temp, self.hparams.n_views)
        self.clu_loss = ClusterLoss(self.hparams.class_num, self.hparams.clu_temp,
                                    self.hparams.n_views)
        self.AEloss = nn.MSELoss()

    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.epochs
    
    def train_dataloader(self):
        return DataLoader(rep_data,
                          batch_size=self.hparams.batch_size, 
                          sampler=SubsetRandomSampler(list(range(self.hparams.train_size))))
        
    def val_dataloader(self):
        return DataLoader(rep_data,
                      batch_size=self.hparams.batch_size, 
                      shuffle=False,
                      sampler=SequentialSampler(list(range(self.hparams.train_size + 1, 
                              self.hparams.train_size + self.hparams.validation_size))))
        
    def forward(self, X):
        return self.model(X)
    
    def step(self, batch, step_name = "train"):
        all_emb = []
        all_clu = []
        
        X, Y = batch
        loss = 0
        if self.hparams.smooth:
            X = preprocess(X)
        ## split to single replicate
        embX, cluX, decodedX = self.forward(X)
        all_emb.append(embX)
        all_clu.append(cluX)
        
        loss += self.hparams.beta * self.AEloss(X, decodedX)
    
        ## compute uniform and align
        uni = 0
        aln = 0
        uni += torch.sum(uniform_loss(normalize(embX.squeeze(1))))

        for i in range(self.hparams.n_rep - 1):
            if self.hparams.n_rep == 2:
                rep = Y
            elif len(Y.size()) == 4: ## with segments (multiple features)
                rep = Y[:, i, :, :].squeeze(1)
            else: ## only coverage
                rep = Y[:, i, :].unsqueeze(1)
                
            if self.hparams.smooth:
                rep = preprocess(rep)

            embY, cluY, decodedY = self.forward(rep)
            loss += self.hparams.beta * self.AEloss(rep, decodedY)
            all_emb.append(embY)
            all_clu.append(cluY)      

        loss /= self.hparams.n_rep
        
        orders = [(a, b) for idx, a in enumerate(range(self.hparams.n_rep)) for b in range(self.hparams.n_rep)[idx + 1:]]
        
        for ords in orders:
            loss_instance = self.ins_loss(all_emb[ords[0]], all_emb[ords[1]])
            loss_cluster = self.clu_loss(all_clu[ords[0]], all_clu[ords[1]])

            loss += loss_instance + loss_cluster
    
        loss /= len(orders)    
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return { ("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                        "progress_bar": {loss_key: loss}}
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")
       
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")
    
    def validation_end(self, outputs):
        if len(outputs) == 0:
            return {"val_loss": torch.tensor(0)}
        else:
            loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            return {"val_loss": loss, "log": {"val_loss": loss}}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.hparams.lr)
        return [optimizer], []
    
class ContrastLearn_lab(pl.LightningModule):
    def __init__(self, hparams):
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        super().__init__()
        self.save_hyperparameters(hparams)
        emb = RegionEmbedding(self.hparams.hidden_size, self.hparams.emb_size, 
                              self.hparams.first_kernel_size, self.hparams.dropout_rate,
                              self.hparams.input_dim)
        self.model = Network(emb, self.hparams.fea_dim, self.hparams.class_num)
        self.ins_loss = InstanceLoss(self.hparams.ins_temp, self.hparams.n_views)
        self.clu_loss = nn.CrossEntropyLoss(weight = self.hparams.class_weights)

    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.epochs
    
    def train_dataloader(self):
        return DataLoader(rep_data,
                          batch_size=self.hparams.batch_size, 
                          sampler=SubsetRandomSampler(list(range(self.hparams.train_size))))
        
    def val_dataloader(self):
        return DataLoader(rep_data,
                      batch_size=self.hparams.batch_size, 
                      shuffle=False,
                      sampler=SequentialSampler(list(range(self.hparams.train_size + 1, 
                              self.hparams.train_size + self.hparams.validation_size))))
        
    def forward(self, X):
        return self.model(X)
    
    def step(self, batch, step_name = "train"):
        X, Y = batch
        Y.unsqueeze(1);
        loss = 0
        rep = X[:, 0, :].unsqueeze(1)
        embX, cluX = self.forward(rep)
        
        loss_cluster = self.clu_loss(cluX.squeeze(1), Y[:, 0])
        loss = loss_cluster
        ## split to single replicate
        for i in range(self.hparams.n_rep-1):
            rep = X[:, i+1, :].unsqueeze(1)
            embY, cluY = self.forward(rep)
            loss_instance = self.ins_loss(embX, embY)
            loss_cluster += self.clu_loss(cluY.squeeze(1), Y[:, i+1])
            loss += loss_instance + loss_cluster
        
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return { ("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                        "progress_bar": {loss_key: loss}}
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

        
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")
    
    def validation_end(self, outputs):
        if len(outputs) == 0:
            return {"val_loss": torch.tensor(0)}
        else:
            loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            return {"val_loss": loss, "log": {"val_loss": loss}}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.hparams.lr)
        return [optimizer], []
   
