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
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

#1/17/2024
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from train import *


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
 
def prepare_dataloader(dataset: Dataset, batch_size: int): #1/18/2024 -  need to check if this function takes the data structure that we have now
                                                            # batch_size is likely something like total number of datapoints/number of gpus
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def train(rank, world_size, args):
#    random.seed(args.seed) 
#    n_rep = args.n_rep

    # Load raw input data
#    datapath = [args.datapath + '/rep' + str(x)  + '.txt'  for x in list(range(1, int(n_rep) + 1))]
#    d = []
#    for file in datapath:
#        if args.debug:
#            print("Reading RCL input file " + file + ".")
#        cov = read_data_new(file)
#        d.append(cov)
     
    # Test set, if we want to test on certain samples only
#    if args.sample != 'null':
#        selected = np.random.choice(d[0].shape[0], int(d[0].shape[0] * 0.85), replace = False)
#        pickle.dump(selected, open(str(args.sample) + ".p", "wb"))
#        d = np.array(d)
#        d = d[:, selected, :]
#        d = list(d)

#    n_dat = len(d[0])
    
    # Initialize distributed environment
    ddp_setup()
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

#    train_data = prepare_dataloader(d, batch_size = args.batch_size_split)
    #n_train = math.ceil(n_dat * 0.8)
    #n_val = n_dat - n_train
    #input_size = len(d[0][0])
    #if args.model == 'ResAE':
    #    input_size = (1, input_size)
    #input_dim = 1

    # If we include fragment length in the training data
    #if len(args.fragpath) > 0:
    #    input_dim = 2
    #    print("Reading fragment length files\n")
    #    for i, f in enumerate(args.fragpath):
    #        fra = read_fragment(f, [1, 2, 3, 4, 6])
    #        d[i] = np.dstack((fra, d[i]))
       
    #w_pos = 0
    #w_neg = 0

    #if args.labpath != 'null':
    #    lab = pickle.load(open(args.labpath, "rb"))
    #    posn = 0
    #    negn = 0
    #    for l in lab:
    #        negn += l.count(0)
    #        posn += l.count(1)
    #    w_pos = (posn + negn) / (2 * posn)
    #    w_neg = (posn + negn) / (2 * negn)
    #    rep_data = combine_rep_lab(d, lab, device=device_id)
    #else:
    #    rep_data = combine_reps(d, device=device_id) #need to change d to the data that got distributed across gpus
    
    #print(rep_data)
    
    
    #class_weights = torch.FloatTensor([w_neg, w_pos]).to(device_id)
    #print("weight ", class_weights)
 #   print("Finished reading\n")

    #hparams = Namespace(lr=args.lr,
    #                    epochs=args.epochs,
    #                    batch_size=args.batch_size,
    #                    batch_size_split=args.batch_size_split,
    #                    train_size=n_train,
    #                    validation_size=n_val,
    #                    hidden_size=args.hidden_size,
    #                    emb_size=args.emb_size,
    #                    fea_dim=args.fea_dim, ## dim of feature for computing loss
    #                    input_dim=input_dim,
    #                    input_size = input_size,
    #                    class_num=args.n_class,
    #                    ins_temp=args.temperature,
    #                    clu_temp=args.temperature,
    #                    n_views=2, ## this is for pairwise comparison
    #                    n_rep = n_rep,
    #                    first_kernel_size = args.first_kernel_size, 
    #                    dropout_rate = args.dropout_rate,
    #                    device = device,
    #                    smooth = args.smooth,
    #                    class_weights = class_weights,
    #                    modelname = args.model,
    #                    beta = 1 ## penalty for encoder decoder, need to tune according to different data
    #                    )

    print(f"Start training on rank {rank}\n")
    #if args.labpath != 'null':
    #    module = ContrastLearn_lab(hparams).to(device_id)
    #else:
    #    module = ContrastLearn(hparams).to(device_id)
    
    # Wrap the model with DistributedDataParallel
    #module = DistributedDataParallel(module, device_ids=[device_id])

    #not sure how to tell this trainer to know exactly what device we are on
    #trainer = pl.Trainer(
    #        accelerator='gpu',
    #        devices=args.gpus,
    #        max_epochs=hparams.epochs,
    #        strategy="ddp"  # Use Distributed Data Parallel
    #        )	# devices -> gpus

    #trainer.fit(module)
    #print(f"Done on {rank}\n")

    #checkpoint_file = args.modelpath
    #trainer.save_checkpoint(checkpoint_file)
    
    
def main(args):
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
    print(n_dat)

    
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
    main(args)
    mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args), join=True)  
