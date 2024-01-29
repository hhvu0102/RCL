import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate
from torch.multiprocessing import cpu_count
from torch.optim import Adam
import pytorch_lightning as pl
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np
import math
import random
import argparse
from argparse import Namespace
import pickle

from architect import *
from ios import read_data_new, read_coverage, read_fragment, combine_reps

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


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
        self.rep_data = self.hparams.rep_data
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = self.model.to(self.gpu_id)
        #self.model = DDP(self.model, device_ids=[self.gpu_id])

    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.epochs
    
    def train_dataloader(self):
        train_data = DataLoader(self.hparams.rep_data,
                                batch_size=self.hparams.batch_size,
                                sampler=SubsetRandomSampler(list(range(self.hparams.train_size))),
                                collate_fn=lambda x: tuple(x_.to(self.gpu_id) for x_ in default_collate(x)) )
        return train_data
      
    def val_dataloader(self):
        val_data = DataLoader(self.hparams.rep_data,
                              batch_size=self.hparams.batch_size,
                              shuffle=False,
                              sampler=SequentialSampler(list(range(self.hparams.train_size + 1,
                                                                   self.hparams.train_size + self.hparams.validation_size))),
                              collate_fn=lambda x: tuple(x_.to(self.gpu_id) for x_ in default_collate(x)) )
        return val_data
        
    def forward(self, X):
        return self.model(X)
    
    def step(self, batch, step_name = "train"):
        all_emb = []
        all_clu = []
     
        print("length of batch")
        print(len(batch))

        print("length of batch[0]")
        print(len(batch[0])) 

        X, Y = batch
        loss = 0
        if self.hparams.smooth:
            X = preprocess(X)
        ## split to single replicate
        embX, cluX, decodedX = self.forward(X)
        all_emb.append(embX)
        all_clu.append(cluX)
        
        loss = loss + self.hparams.beta * self.AEloss(X, decodedX)
    
        ## compute uniform and align
        uni = 0
        aln = 0
        uni = uni + torch.sum(uniform_loss(normalize(embX.squeeze(1))))

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
            loss = loss + self.hparams.beta * self.AEloss(rep, decodedY)
            all_emb.append(embY)
            all_clu.append(cluY)      

        loss = loss / self.hparams.n_rep
        
        orders = [(a, b) for idx, a in enumerate(range(self.hparams.n_rep)) for b in range(self.hparams.n_rep)[idx + 1:]]
        
        for ords in orders:
            loss_instance = self.ins_loss(all_emb[ords[0]], all_emb[ords[1]])
            loss_cluster = self.clu_loss(all_clu[ords[0]], all_clu[ords[1]])

            loss = loss + loss_instance + loss_cluster
    
        loss = loss / len(orders)    
        loss_key = f"{step_name}_loss_{self.gpu_id}"
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
   
def ddp_setup():
    dist.init_process_group(backend="gloo")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main(args):
     # Initialize distributed environment
    ddp_setup()
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count() 
    print(device_id)

    # Set up
    random.seed(args.seed) 
    n_rep = args.n_rep

    # Load raw input data
    datapath = [args.datapath + '/rep' + str(x)  + '.txt'  for x in list(range(1, int(n_rep) + 1))]
    d = []
    for file in datapath:
        if args.debug:
            print("Reading RCL input file " + file + ".")
        cov = read_data_new(file)
        d.append(cov)
     
    # Test set, if we want to test on certain samples only
    if args.sample != 'null':
        selected = np.random.choice(d[0].shape[0], int(d[0].shape[0] * 0.85), replace = False)
        pickle.dump(selected, open(str(args.sample) + ".p", "wb"))
        d = np.array(d)
        d = d[:, selected, :]
        d = list(d)

    n_dat = len(d[0]) 

    print(f"Start on rank {rank}\n")
    
    n_train = math.ceil(n_dat * 0.8) #/ torch.cuda.device_count())
    n_val = n_dat - n_train # math.floor(n_dat / torch.cuda.device_count()) - n_train
    

    input_size = len(d[0][0]) #here
    if args.model == 'ResAE':
        input_size = (1, input_size)
    input_dim = 1

    # If we include fragment length in the training data
    if len(args.fragpath) > 0:
        input_dim = 2
        print("Reading fragment length files\n")
        for i, f in enumerate(args.fragpath):
            fra = read_fragment(f, [1, 2, 3, 4, 6])
            d[i] = np.dstack((fra, d[i]))
       
    w_pos = 0
    w_neg = 0

    if args.labpath != 'null':
        lab = pickle.load(open(args.labpath, "rb"))
        posn = 0
        negn = 0
        for l in lab:
            negn += l.count(0)
            posn += l.count(1)
        w_pos = (posn + negn) / (2 * posn)
        w_neg = (posn + negn) / (2 * negn)
        rep_data = combine_rep_lab(d, lab, device=device_id)
    else:
        rep_data = combine_reps(d, device=device_id)

    
    print(len(rep_data))
    print(len(rep_data[0]))
    
    class_weights = torch.FloatTensor([w_neg, w_pos]).to(device_id)
    print("weight ", class_weights)
    print("Finished reading\n")
    
    random.seed(args.seed)
    hparams = Namespace(rep_data=rep_data,
                        lr=args.lr,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        batch_size_split=args.batch_size_split,
                        train_size=n_train,
                        validation_size=n_val,
                        hidden_size=args.hidden_size,
                        emb_size=args.emb_size,
                        fea_dim=args.fea_dim, ## dim of feature for computing loss
                        input_dim=input_dim,
                        input_size = input_size,
                        class_num=args.n_class,
                        ins_temp=args.temperature,
                        clu_temp=args.temperature,
                        n_views=2, ## this is for pairwise comparison
                        n_rep = n_rep,
                        first_kernel_size = args.first_kernel_size, 
                        dropout_rate = args.dropout_rate,
                        device = "cuda:" + str(device_id),
                        smooth = args.smooth,
                        class_weights = class_weights,
                        modelname = args.model,
                        beta = 1 ## penalty for encoder decoder, need to tune according to different data
                        )

    print(f"Start training on rank {rank}\n")
    if args.labpath != 'null':
        module = ContrastLearn_lab(hparams).to(device_id)
    else:
        module = ContrastLearn(hparams).to(device_id)
   
#    trainer = pl.Trainer(gpus=args.gpus, max_epochs=hparams.epochs) 
    trainer = pl.Trainer(
            accelerator='gpu',
            devices = torch.cuda.device_count(), #args.gpus,
            max_epochs=hparams.epochs,
            strategy="ddp"  # Use Distributed Data Parallel
            )	# devices -> gpus

    trainer.fit(module)
    print(f"Done on {rank}\n")

    checkpoint_file = args.modelpath
    trainer.save_checkpoint(checkpoint_file)
    
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--n_rep', type=int)
    parser.add_argument('--fragpath', default=[], nargs='*')
    parser.add_argument('--modelpath', default='model.ckpt', type=str)
    parser.add_argument('--labpath', default='null', type=str)
    parser.add_argument('--model', default='ResAE', type=str) ## model name, can be ResAE, Resnet and CNNAE
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--batch_size_split', default=1000, type=int) #size data to put in each GPU. Need to be updated when adding --distributed option
    parser.add_argument('--emb_size', default=50, type=int) ## notice the dim after cov is 80
    parser.add_argument('--fea_dim', default=25, type=int)
    parser.add_argument('--first_kernel_size', default=31, type=int)
    parser.add_argument("--dropout_rate", default=0.05, type=float)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--hidden_size", default=5, type=int) ## num of hidden channels
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--smooth", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--sample", default='null', type=str)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    main(args)

    #mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args), join=True)  
    
    
   
    
    
    
    
    
    
    
    
    

