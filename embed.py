import numpy as np
import torch
import torch.nn as nn
import sys
from conclu import *
from main import *
from ios import *
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
from sklearn.metrics import auc
import os
import argparse
import pandas as pd
from functools import reduce
import warnings


class Embed(nn.Module):
    def __init__(self, embeddings_model_path):
        super().__init__()
        
        base_model = ContrastLearn.load_from_checkpoint(embeddings_model_path).model
        
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.instance_projector = base_model.instance_projector
        self.unflat = base_model.unflat
        
    def forward(self, x, *args):
        for res_block in self.encoder:
            x = res_block(x)
        h = self.instance_projector(x)
#        x = self.unflat(h)
        decoded = self.decoder(x)
        return h, decoded
    

def get_conemb(pos, func):
    pos_tensor = []
    for t in pos:
        pos_tensor.append(torch.from_numpy(t).float().unsqueeze(0).unsqueeze(0))
    pos_tensor = torch.cat(pos_tensor)
    
    if len(pos_tensor.size()) == 4:
        pos_tensor = pos_tensor.squeeze(1)
    pos = func(pos_tensor)
    return pos   

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='metric', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str)
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--start_rep", type=int)
    parser.add_argument("--stop_rep", type=int)

    args = parser.parse_args()
    mod = args.model
    #n_rep = args.n_rep
    datapath = [args.datapath + '/rep' + str(x)  + '.txt'  for x in list(range(int(args.start_rep), int(args.stop_rep) + 1))]
    embed = Embed(mod)  

    dat = []
    for file in datapath:
        cov = read_data_new(file)
        dat.append(cov)

    rep_emb = []
    for d in dat:
        rep_emb.append(get_conemb(d, embed))

    c = int(args.start_rep)
    print("Saving embedding data")
    for rep in rep_emb:
        m = rep[1].squeeze(1).detach().numpy()
        np.savetxt(args.datapath + '/rep' + str(c) + '.embed', m, fmt='%.2f')
        c += 1
    
    c = int(args.start_rep)
    print("Saving raw data")
    for d in dat:
        np.savetxt(args.datapath + '/rep' + str(c) + '.raw', d, fmt='%.2f')
        c += 1
