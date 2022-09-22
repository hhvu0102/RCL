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

class Classify(nn.Module):
    def __init__(self, embeddings_model_path):
        super().__init__()
        
        base_model = ContrastLearn.load_from_checkpoint(embeddings_model_path).model
        
        self.encoder = base_model.encoder
        self.instance_projector = base_model.instance_projector
        self.clustering = base_model.cluster_projector
        
    def forward(self, x, *args):
        for res_block in self.encoder:
            x = res_block(x)
        h = self.instance_projector(x)
        h = h.unsqueeze(1)
        print(h.size())
        ass = self.clustering(h)    
        return ass

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
#         x = self.unflat(h)
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

def compute_m(dat, rep_class, alllab2):
    selected = 0
    m = alllab2[0]
    final_lab = []
    for d in rep_class:
        lab1 = torch.argmax(d.squeeze(1), dim=1).tolist()
        f1 = f1_score(m, lab1)

        lab2 = torch.argmin(d.squeeze(1), dim=1).tolist()
        f2 = f1_score(m, lab2)

        if f2 > f1:
            lab = lab2
        else:
            lab = lab1

#         print("peak number ", lab.count(1))
        final_lab.append(lab)

    return final_lab

def make_region(datapath):
    df = pd.read_csv(datapath, header = None, sep = "\t", names=["chr", "s", "e", "name", "conut"])
    a = df.groupby("name", sort=False).s.idxmin()
    b = df.groupby("name", sort=False).e.idxmax()
    d = df.loc[a]
    d['e'] =  df.loc[b]['e'].values
    d = d[d.columns[1:4]]
    d = d.reset_index(drop=True)
    return d


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='metric', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dpath", default=[], nargs='*')
    parser.add_argument("--prefix", type=str) # output path
    parser.add_argument("--id", type=int)
    parser.add_argument("--psudo", type=int, default = 1)

    args = parser.parse_args()

    #classfication = Classify('chr' + str(args.id) + "/" + str(args.model)) ## this is training seperately
    classfication = Classify(str(args.model))    ## train on all (80 or 90 %)
    #datapath = ['chr' + str(args.id) + "/" + str(args.id) + ".MCF7_" + str(i) + ".covBga-noBl.txt" for i in range(47, 49)]
    datapath = args.dpath
    dat = []
    dataf = []
    for i in datapath:
        dat.append(read_data_new(i))
        dataf.append(make_region(i))

    alllab2 = []
    if args.psudo:
        for d in dat:
            tmp = np.sum(d, axis = 1)
            alllab2.append(tmp>30000) # this is just a rough threshold for determing peak
    else:
        alllab2 = pickle.load(open('chr' + str(args.id) + "/chip_nbl.p", "rb" )) # not applicable 

    rep_class = []
    for d in dat:
        rep_class.append(get_conemb(d, classfication))

    final_lab = compute_m(dat, rep_class, alllab2)
    i = 1
    y_true=alllab2[0]

    for p, f, o in zip(rep_class, final_lab, dataf):
        p = p.squeeze(1).detach().cpu().numpy()
        y_scores = p[:, 0]
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        auc1 = auc(recall, precision)
        y_scores = p[:, 1]
        precision2, recall2, thresholds = precision_recall_curve(y_true, y_scores)
        auc2 = auc(recall2, precision2)
        if auc2 > auc1:
            #print(auc2, recall2, precision2)
            y_scores = p[:, 1]
        else:
            #print(auc1, recall, precision)
            y_scores = p[:, 0]

        d = {'chr': str(args.id), 'score': y_scores, 'pred': f}
        df = pd.DataFrame(data=d)
        df = pd.concat([df, o], axis=1)[["chr", "s", "e", "name", "score", "pred"]]
        df.to_csv(str(args.prefix) + '/rcl' + '.' + str(i) + '.score', index=False)
        i += 1



