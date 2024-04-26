import os
import copy
import psutil
from pathlib import Path
import pdb
from tqdm import tqdm
import os
import sys
import argparse
import logging 
from logging import debug as DEBUG
from logging import info as INFO
logging.basicConfig(level=0,format='[log: %(filename)s line:%(lineno)d] %(message)s')
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import random
import pickle
from numpy import exp, sqrt, dot
from scipy.spatial.distance import cdist
from scipy.io import loadmat
from random import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import moment
from scipy.stats import skew
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchmetrics
import csv
from operator import truediv

from module import Attention, Block

def concatdata(startyear,endyear,loaddir,usevar,endday):
# startyear=2008
# endyear=2017
# loaddir='dataset/NPY/'
    # DEBUG(f'startyear: {startyear} endyear: {endyear}')
    X=[]
    y=[]
    for year in list(range(startyear,endyear+1)):
        # DEBUG(f"year: {year}")
        file_name_X = loaddir+'X_All_'+str(year)+'_endday_278.npy'
        file_name_y = loaddir+'y_All_'+str(year)+'_endday_278.npy'
        # DEBUG(f"file_name_X: {file_name_X}")
        # DEBUG(f"file_name_y: {file_name_y}")
        data_X = np.load(file_name_X, allow_pickle=True)
        data_y = np.load(file_name_y, allow_pickle=True)
        # DEBUG(f"data_X: {data_X.shape}")
        # DEBUG(f"data_y: {data_y.shape}")
        X.append(data_X)
        y.append(data_y)
        # pdb.set_trace()

    Xset=np.concatenate(X)
    yset=np.concatenate(y)
    # DEBUG(f"Xset:{Xset.shape}")
    # DEBUG(f"yset:{yset.shape}")
    return Xset, yset

def feature_normalize_np(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return truediv((data - mu),std)


def R2andRMSE(all_pre,all_label):
    # DEBUG(all_pre.shape)
    # DEBUG(all_label.shape)
    all_label=np.reshape(all_label,-1)
    all_pre=np.reshape(all_pre,-1)
    all_pre=torch.tensor(all_pre)
    all_label=torch.tensor(all_label)
    
    rmse=torchmetrics.MeanSquaredError(squared=False)
    rmse_value=rmse(all_pre,all_label)
    R2score=torchmetrics.R2Score()
    R2score_value=R2score(all_pre,all_label)
    return R2score_value.cpu().data.item(),rmse_value.cpu().data.item()
    
def isplit_by_n(ls, n):
    for i in range(0, len(ls), n):
        yield ls[i:i+n]

def split_by_n(ls, n):
    return list(isplit_by_n(ls, n))

def instance_to_bag(inspreds,n=100):
    DEBUG(f"inspreds: {inspreds.shape}")
    inspreds_list = inspreds.tolist()
    DEBUG(f"inspreds: {len(inspreds)}")
    composelist = []
    for i in range(0, len(inspreds_list), n):
        composelist.append(inspreds_list[i:i+n])
    # composelist=split_by_n(inspreds_list,n)
    DEBUG(f"composelist: {len(composelist)} {len(composelist[0])}")
    # pdb.set_trace()
    composelist = np.array(composelist)
    DEBUG(f"composelist: {composelist.shape}")
    composelist_mean = np.mean(composelist, axis=1)
    DEBUG(f"composelist_mean: {composelist_mean.shape}")
    # pdb.set_trace()
    return composelist_mean

class AttentionMLPRegressor(nn.Module):
    def __init__(self, args):
        super(AttentionMLPRegressor, self).__init__()
        self.args = args
        self.query = nn.Linear(args.hidden_size, args.hidden_size)
        self.key = nn.Linear(args.hidden_size, args.hidden_size)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)
        self.attn = nn.MultiheadAttention(args.hidden_size, args.num_heads, batch_first=True)
        self.fc1 = nn.Linear(args.hidden_size, 1)
        self.fc2 = nn.Linear(args.N, 1)
    def forward(self, x):
        DEBUG(f'x: {x.shape}')
        q = self.query(x)
        DEBUG(f'q: {q.shape}')
        k = self.key(x)
        DEBUG(f'k: {k.shape}')
        v = self.value(x)
        DEBUG(f'v: {v.shape}')
        x, att = self.attn(q, k, v)
        DEBUG(f'x: {x.shape}')
        DEBUG(f'att: {att.shape}')
        x = self.fc1(x)
        DEBUG(f'x: {x.shape}')
        x = x.squeeze(2)
        DEBUG(f"x: {x.shape}")
        x = self.fc2(x)
        DEBUG(f'x: {x.shape}')
        # pdb.set_trace()
        return x, att




class MIRDataset(data_utils.Dataset):
    def __init__(self, args, data, label):
        self.data = data
        self.label = label
        DEBUG(f"self.data: {self.data.shape}")
        DEBUG(f"self.label: {self.label.shape}")
        DEBUG(f'args.N: {type(args.N)} {args.N}')
        self.databag = []
        self.labelbag = []
        for i in range(0, len(self.data), args.N):
            self.databag.append(self.data[i:i+args.N])
            self.labelbag.append(self.label[i:i+args.N])
        DEBUG(f"self.databag: {len(self.databag)}")
        DEBUG(f"self.labelbag: {len(self.labelbag)}")
        # pdb.set_trace()
    def __len__(self):
        return len(self.databag)

    def __getitem__(self, index):
        x = self.databag[index]
        y = self.labelbag[index]
        return x, y

def val(args, model, test_x, test_y):   
    model.eval() 
    pred, gt, att = test(args, model, test_x, test_y)
    DEBUG(f"pred: {pred.shape}")
    DEBUG(f"gt: {gt.shape}")
    DEBUG(f"att: {att.shape}")
    
    y_bag = instance_to_bag(test_y,n=args.N)
    DEBUG(f"y_bag: {y_bag.shape}")
    DEBUG(f"gt: {gt[:10]}")
    DEBUG(f"y_bag: {y_bag[:10]}")
    
    pred=pred.reshape(-1,1)
    y_bag=y_bag.reshape(-1,1)
    DEBUG(f"pred: {pred.shape}")
    DEBUG(f"y_bag: {y_bag.shape}")

    r2_val,rmse_val=R2andRMSE(pred,y_bag)
    DEBUG(f"r2_val: {r2_val}")
    DEBUG(f"rmse_val: {rmse_val}")
    # pdb.set_trace()
    return r2_val, rmse_val, pred, y_bag, att

def get_params(optimizer):
    for param_group in optimizer.param_groups:
        return param_group
    
def train(args, model, train_x, train_y, test_x, test_y):
    # data = MIRDataset(args, train_x, train_y)
    # DEBUG(f"data: {len(data)}")
    # for idx, (databag, labelbag) in enumerate(data):
    #     DEBUG(f"idx: {idx} databag: {databag.shape} labelbag: {labelbag.shape}")
    #     pdb.set_trace()

    DEBUG(f"train_x: {train_x.shape}")
    DEBUG(f"train_y: {train_y.shape}")
    split = int(len(train_x)//args.N*0.8)*args.N
    DEBUG(f"split: {split}")
    train_x, val_x = train_x[:split], train_x[split:]
    train_y, val_y = train_y[:split], train_y[split:]
    DEBUG(f"train_x: {train_x.shape}")
    DEBUG(f"train_y: {train_y.shape}")
    DEBUG(f"val_x: {val_x.shape}")
    DEBUG(f"val_y: {val_y.shape}")
    # pdb.set_trace()
    train_loader = data_utils.DataLoader(MIRDataset(args, train_x, train_y),
                                         batch_size=1,
                                         shuffle=True)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience = 3, verbose=True)
    creterion = nn.MSELoss()
    model.train()
    model.to(args.device)
    rmse_best = 1000000
    r2_best = -100000
    pred_best = None
    y_best = None
    att_best = None
    model_best = None
    stop = 0
    for epoch in range(args.epoch):
        batch_total = 0
        loss_total = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            # logging.getLogger().setLevel(logging.INFO)
            DEBUG(f"batch_idx: {batch_idx} data: {data.shape} label: {label.shape}")
            data = data.to(args.device)
            label = label.to(args.device)
            data = data.to(torch.float32)
            label = label.to(torch.float32)
            label = torch.mean(label, dim=1)
            DEBUG(f"label: {label.shape}")
            # pdb.set_trace()
            target, att = model(data)
            DEBUG(f"target: {target.shape}")
            DEBUG(f"att: {att.shape}")

            loss = creterion(target, label)
            DEBUG(f"loss: {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            batch_total += 1
            
            # break
            # pdb.set_trace()

        loss_info = loss_total / batch_total
        r2_val, rmse_val, pred_val, y_val, att_val = val(args, model, val_x, val_y)
        if(epoch % args.train_info == 0) and (epoch != 0):
            INFO(f"Epoch: {epoch}\{args.epoch} lr: {get_params(optimizer)['lr']:.5f} stop: {stop} loss_info: {loss_info:.8f}")
            INFO(f"Validation: R2: {r2_val:.5f} RMSE: {rmse_val:.5f}")

        scheduler.step(rmse_val)

        if(rmse_val < rmse_best):
            rmse_best = rmse_val
            r2_best = r2_val
            pred_best = pred_val
            y_best = y_val
            att_best = att_val
            model_best = copy.deepcopy(model)
            stop = 0
        else:
            stop += 1

        # if(loss_info < loss_result):
        #     loss_result = loss_info
        #     stop = 0
        # else:
        #     stop += 1

        if(stop > 10):
            INFO(f"best val result r2: {r2_best:.5f} rmse: {rmse_best:.5f}")
            break
    r2_test, rmse_test, pred_test, y_test, att_test = val(args, model_best, test_x, test_y)
    rmse_best = rmse_test
    r2_best = r2_test
    pred_best = pred_test
    y_best = y_test
    att_best = att_test
    INFO(f"test result r2: {r2_best:.5f} rmse: {rmse_best:.5f}")
    
    return r2_best, rmse_best, pred_best, y_best, att_best


def test(args, model, test_x, test_y):
    # data = MIRDataset(train_x, train_y)
    # DEBUG(f"data: {len(data)}")
    # for idx, (databag, labelbag) in enumerate(data):
    #     DEBUG(f"idx: {idx} databag: {databag.shape} labelbag: {labelbag.shape}")
    #     pdb.set_trace()
    test_loader = data_utils.DataLoader(MIRDataset(args, test_x, test_y),
                                         batch_size=1,
                                         shuffle=False)
    model.eval()
    model.to(args.device)
    pred_list = []
    gt_list = []
    att_list = []
    creterion = nn.MSELoss()
    for batch_idx, (data, label) in enumerate(test_loader):
        # logging.getLogger().setLevel(logging.INFO)
        DEBUG(f"batch_idx: {batch_idx} data: {data.shape} label: {label.shape}")
        data = data.to(args.device)
        label = label.to(args.device)
        data = data.to(torch.float32)
        label = label.to(torch.float32)
        label = torch.mean(label, dim=1)
        DEBUG(f"label: {label.shape}")
        # pdb.set_trace()
        target, A = model(data)
        DEBUG(f"target: {target.shape}")
        DEBUG(f"A: {A.shape}")
        loss = creterion(target, label)
        
        target = target.detach().cpu()
        target = target.numpy().tolist()
        DEBUG(f"target: {target}")
        pred_list.append(target)

        label = label.detach().cpu()
        label = label.numpy().tolist()
        DEBUG(f"label: {label}")
        gt_list.append(label)

        att = A.detach().cpu()
        att = att.numpy()
        DEBUG(f"att: {att}")
        att_list.append(att)
        # pdb.set_trace()

    DEBUG(f"pred_list: {len(pred_list)}")
    pred_list = np.array(pred_list)
    pred_list = pred_list.reshape(-1, 1)
    DEBUG(f"pred_list: {pred_list.shape}")

    gt_list = np.array(gt_list)
    gt_list = gt_list.reshape(-1, 1)
    DEBUG(f"gt_list: {gt_list.shape}")

    att_list = np.array(att_list)
    DEBUG(f"att_list: {att_list.shape}")
    # pdb.set_trace()
    return pred_list, gt_list, att_list

class AttMIR():
    def __init__(self, args) -> None:
        self.args = args

    def transform(self, ytrain, ytest):
        data = np.concatenate((ytrain, ytest), axis=0)
        scaler1 = StandardScaler()
        scaler1.fit(data)
        ytrain = scaler1.transform(ytrain)
        ytest = scaler1.transform(ytest)

        data = np.concatenate((ytrain, ytest), axis=0)
        scaler2 = MinMaxScaler()
        scaler2.fit(data)
        ytrain = scaler2.transform(ytrain)
        ytest = scaler2.transform(ytest)

        return ytrain, ytest

    def drop_feature_by_usevar(self, data, usevar):
        if(usevar == 'All'):
            return data
        
        # NDWI GCI EVI LSTday LSTnight ppt tmax tmean tmin vpdmax vpdmin awc cec som historical year
        new_data = []
        DEBUG(f"data: {data.shape} {data.size}")
        usevar_index_dict = {'NDWI':0, 'GCI':1, 'EVI':2, 'LSTday':3, 'LSTnight':4, 'ppt':5, 'tmax':6, 'tmean':7, 'tmin':8, 'vpdmax':9, 'vpdmin':10,\
                             'awc':11, 'cec':12, 'som':13, 'historical':14, 'year':15}
        if(usevar not in usevar_index_dict.keys()):
            raise ValueError(f"usevar: {usevar} not in usevar_index_dict.keys(): {usevar_index_dict.keys()}")

        index = usevar_index_dict[usevar]
        if(index<=10):        
            DEBUG(f'check: {data[1, 14*(index+1):14*(index+2)]}')
            mask = []
            for i in range(11):
                if(i == index):
                    mask += [False] * 14
                else:    
                    mask += [True] * 14
            mask += [True] * 5
            DEBUG(f'mask: {mask}')
            new_data = data[:, np.tile(mask, data.shape[1]//len(mask))]
            DEBUG(f'check: {new_data[1, 14*index:14*(index+1)]}')
            self.args.hidden_size = 145
        else:
            DEBUG(f'check: {data[1, 154:159]}')
            mask = [True] * 154
            for i in range(5):
                if(i == index-11):
                    mask += [False] 
                else:    
                    mask += [True] 
            DEBUG(f'mask: {mask}')
            new_data = data[:, np.tile(mask, data.shape[1]//len(mask))]
            DEBUG(f'check: {new_data[1, 154:158]}')
            self.args.hidden_size = 158

        DEBUG(f"new_data: {new_data.shape} {new_data.size}")
        # pdb.set_trace()
        return new_data
        
    def drop_feature_by_endday(self, data, endday):
        if(endday == 278):
            return data
        # 134 150 166 182 198 214 230 246 262 
        new_data = []
        DEBUG(f"data: {data.shape} {data.size}")
        endday_index_dict = {134:5, 150:6, 166:7, 182:8, 198:9, 214:10, 230:11, 246:12, 262:13}
        if(endday not in endday_index_dict.keys()):
            raise ValueError(f"endday: {endday} not in usevar_index_dict.keys(): {endday_index_dict.keys()}")

        index = endday_index_dict[endday]
        for i in range(3):
            DEBUG(f'check: {data[1, 14*i:14*(i+1)]}')
        mask = []
        for i in range(11):
            mask += [True] * index
            mask += [False] * (14-index)
        mask += [True] * 5
        DEBUG(f'mask: {mask}')
        new_data = data[:, np.tile(mask, data.shape[1]//len(mask))]
        for i in range(3):
            DEBUG(f'check: {new_data[1, index*i:index*(i+1)]}')
        self.args.hidden_size = index * 11 + 5
        DEBUG(f"new_data: {new_data.shape} {new_data.size}")
        # pdb.set_trace()
        return new_data

        

    def run_loop_enddays(self):
        for endday in self.args.enddays:
            endday_feature_dict = {134: 60, 150: 71, 166: 82, 182: 93, 198: 104, 214: 115, 230: 126, 246: 137, 262: 148, 278: 159}
            self.args.hidden_size = endday_feature_dict[endday]
            self.run_loop_usevars(endday)
        
    def run_loop_usevars(self, endday):
        for usevar in self.args.usevars:
            self.run(endday, usevar)

    def run(self, endday, usevar):    
        for testyear in args.testyears:
            INFO(f"endday: {endday} usevar: {usevar} testyear: {testyear}")

            train_x,train_y=concatdata(2008,testyear-1,self.args.input_root,usevar,endday)
            DEBUG(f"train_x: {train_x.shape}")
            DEBUG(f"train_y: {train_y.shape}")
            # pdb.set_trace()
            test_x,test_y=concatdata(testyear,testyear,self.args.input_root,usevar,endday)
            DEBUG(f"test_x: {test_x.shape}")
            DEBUG(f"test_y: {test_y.shape}")
            # pdb.set_trace()

            train_x = self.drop_feature_by_usevar(train_x, usevar)
            test_x = self.drop_feature_by_usevar(test_x, usevar)
            DEBUG(f"train_x: {train_x.shape}")
            DEBUG(f"test_x: {test_x.shape}")

            train_x = self.drop_feature_by_endday(train_x, endday)
            test_x = self.drop_feature_by_endday(test_x, endday)
            DEBUG(f"train_x: {train_x.shape}")
            DEBUG(f"test_x: {test_x.shape}")
            # pdb.set_trace()
            train_x=np.reshape(train_x,(-1,self.args.hidden_size))###(N*100,46)
            test_x=np.reshape(test_x,(-1,self.args.hidden_size))###(N*100,46)
            DEBUG(f"reshape train_x: {train_x.shape} {np.max(train_x):.3f} {np.min(train_x):.3f}")
            DEBUG(f"reshape test_x: {test_x.shape} {np.max(test_x):.3f} {np.min(test_x):.3f}")

            train_x, test_x = self.transform(train_x, test_x)
            DEBUG(f"transform train_x: {train_x.shape} {np.max(train_x):.3f} {np.min(train_x):.3f}")
            DEBUG(f"transform test_x: {test_x.shape} {np.max(test_x):.3f} {np.min(test_x):.3f}")

            train_y=np.repeat(train_y,self.args.N,axis=0) ####(N*100,1)
            test_y=np.repeat(test_y,self.args.N,axis=0) ####(N*100,1)
            DEBUG(f"train_y: {train_y.shape} {np.max(train_y):.3f} {np.min(train_y):.3f}")
            DEBUG(f"test_y: {test_y.shape} {np.max(test_y):.3f} {np.min(test_y):.3f}")
            # pdb.set_trace()

            model = AttentionMLPRegressor(self.args)
            output_dir = self.args.output_root+"/"+str(testyear)+"/"+str(endday)+"/"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            r2_best, rmse_best, pred_best, y_best, att_best = train(self.args, model, train_x, train_y, test_x, test_y)

            if(self.args.visualize):
                np.save(output_dir+"/pred_best.npy",pred_best)
                np.save(output_dir+"/y_best.npy",y_best)
                np.save(output_dir+"/att_best.npy",att_best)
            # pdb.set_trace()
                       
                    
if __name__ == '__main__':
    logging.getLogger().setLevel(20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default="", help="The input .npy file directory")
    parser.add_argument('--output_root', type=str, default="", help="The output result directory")
    parser.add_argument('--seed', type=int, default=3407, help="The random seed")
    parser.add_argument('--N', type=int, default=1000, help="The number of instances in a bag")
    parser.add_argument('--device', type=str, default="cpu", help="The device to run the model")
    parser.add_argument('--lr', type=float, default=0.001, help="The learning rate")
    parser.add_argument('--epoch', type=int, default=500, help="The number of epoch")
    parser.add_argument('--train_info', type=int, default=500, help="The number of epoch to print the training information")
    parser.add_argument('--num_heads', type=int, default=12, help="The number of heads in multi-head attention")
    parser.add_argument('--hidden_size', type=int, default=159, help='60, 71, 82, 93, 104, 115, 126, 137, 148, 159')
    parser.add_argument('--visualize', action='store_true', help='visualize the attention map')
    parser.add_argument('--testyears', nargs='+', default='', help='2018 2019 2020 2021 2022')
    parser.add_argument('--enddays', nargs='+', default='', help='134 150 166 182 198 214 230 246 262 278')
    parser.add_argument('--usevars', nargs='+', default='', help='The variables to use: NDWI GCI EVI LSTday LSTnight ppt tmax tmean tmin vpdmax vpdmin awc cec som historical year All')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INFO(f"device: {args.device}")
    args.testyears = [int(i) for i in args.testyears]
    args.enddays = [int(i) for i in args.enddays]

    INFO(f'args:-----------------------------')
    for k,v in vars(args).items():
        INFO(f"{k}: {v}")
    INFO(f'----------------------------------')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    attmir = AttMIR(args)
    attmir.run_loop_enddays()