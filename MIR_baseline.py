import os
import argparse
import pdb
import shutil
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import logging 
from logging import debug as DEBUG
from logging import info as INFO
logging.basicConfig(level=20,format='[log: %(filename)s line:%(lineno)d] %(message)s')
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import moment
from scipy.stats import skew
import torch
import torch.nn as nn
import torchmetrics
import csv
from operator import truediv

def concatdata(startyear,endyear,loaddir,usevar,endday):
# startyear=2008
# endyear=2017
# loaddir='dataset/NPY/'
    X=[]
    y=[]
    for year in list(range(startyear,endyear+1)):
        # DEBUG(f"year: {year}")
        file_name_X = loaddir+'X_'+str(usevar)+'_'+str(year)+'_endday_278.npy'
        file_name_y = loaddir+'y_'+str(usevar)+'_'+str(year)+'_endday_278.npy'
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

def drawfigures(args, pre,label,outstring,name,testyear):
    x=label
    y=pre
    plt.ylabel('predicted')
    plt.xlabel('observed')
    plt.text(min(min(x),min(y)),max(max(x),max(y)),outstring)
    plt.scatter(x, y)
    plt.plot([min(min(x),min(y)),max(max(x),max(y))],[min(min(x),min(y)),max(max(x),max(y))], 'k--')
    plt.title(name)
    plt.savefig(args.output_root+'/Figures/'+str(testyear)+name+'.pdf',bbox_inches = 'tight',dpi=960)
    plt.close()
    
def isplit_by_n(ls, n):
    for i in range(0, len(ls), n):
        yield ls[i:i+n]

def split_by_n(ls, n):
    return list(isplit_by_n(ls, n))

def instance_to_bag(inspreds,n=100):
    DEBUG(f"inspreds:\n{inspreds[:10]}")
    composelist=split_by_n(inspreds.tolist(),n)
    DEBUG(f"composelist: {len(composelist)}")
    DEBUG(f"{composelist[0][:10]}")
    return np.mean(np.array(composelist),axis=1)


class MIRBASE():
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
            self.run_loop_usevars(endday)
        
    def run_loop_usevars(self, endday):
        for usevar in self.args.usevars:
            self.run(endday, usevar)

    def run(self,endday, usevar):
            for testyear in args.testyears:
                INFO(f"endday: {endday} usevar: {usevar} testyear: {testyear}")
                output_dir = self.args.output_root+"/"+str(testyear)+"/"+str(endday)+"/"
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                train_x,train_y=concatdata(2008,testyear-1,self.args.input_root,usevar,endday)
                DEBUG(f"train_x: {train_x.shape}")
                DEBUG(f"train_y: {train_y.shape}")
                # pdb.set_trace()
                test_x,test_y=concatdata(testyear,testyear,self.args.input_root,usevar,endday)
                DEBUG(f"test_x: {test_x.shape}")
                DEBUG(f"test_y: {test_y.shape}")
                
                train_x = self.drop_feature_by_endday(train_x, endday)
                test_x = self.drop_feature_by_endday(test_x, endday)
                DEBUG(f"train_x: {train_x.shape}")
                DEBUG(f"test_x: {test_x.shape}")

                dimlength=train_x.shape[1]//self.args.N
                train_x=np.reshape(train_x,(-1,dimlength))###(N*100,46)
                test_x=np.reshape(test_x,(-1,dimlength))###(N*100,46)
                DEBUG(f"train_x: {train_x.shape} {np.max(train_x):.3f} {np.min(train_x):.3f}")
                DEBUG(f"test_x: {test_x.shape} {np.max(test_x):.3f} {np.min(test_x):.3f}")

                train_x, test_x = self.transform(train_x, test_x)
                DEBUG(f"train_x: {train_x.shape} {np.max(train_x):.3f} {np.min(train_x):.3f}")
                DEBUG(f"test_x: {test_x.shape} {np.max(test_x):.3f} {np.min(test_x):.3f}")
                transformedtrain_y=np.repeat(train_y,self.args.N,axis=0) ####(N*100,1)
                transformedtest_y=np.repeat(test_y,self.args.N,axis=0) ####(N*100,1)
                DEBUG(f"transformedtrain_y: {transformedtrain_y.shape} {np.max(transformedtrain_y):.3f} {np.min(transformedtrain_y):.3f}")
                DEBUG(f"transformedtest_y: {transformedtest_y.shape} {np.max(transformedtest_y):.3f} {np.min(transformedtest_y):.3f}")
                # pdb.set_trace()
                
                if(self.args.model=='mlp'):
                    model = MLPRegressor(hidden_layer_sizes=(128,64), learning_rate_init=0.001, max_iter=300, alpha=3,verbose=False)
                if(self.args.model=='lr'):
                    model = LinearRegression()
                if(self.args.model=='ridge'):
                    model = Ridge(alpha=1)
                if(self.args.model=='randomforest'):
                    model = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=0)
                
                model.fit(train_x, transformedtrain_y)
            
                transed_test_pred = model.predict(test_x)
                DEBUG(f"transed_test_pred: {transed_test_pred.shape}")

                pred_best=instance_to_bag(transed_test_pred,n=self.args.N)
                y_best=instance_to_bag(transformedtest_y,n=self.args.N)
                DEBUG(f"pred_best: {pred_best.shape}")
                DEBUG(f"y_best: {y_best.shape}")
                
                pred_best=pred_best.reshape(-1,1)
                y_best=y_best.reshape(-1,1)
                DEBUG(f"pred_best: {pred_best.shape}")
                DEBUG(f"y_best: {y_best.shape}")
                
                if(self.args.visualize):
                    np.save(output_dir+"/pred_best.npy",pred_best)
                    np.save(output_dir+"/y_best.npy",y_best)

                test_R2score_value,test_rmse_value=R2andRMSE(pred_best,y_best)
                DEBUG(f"test_R2score_value: {test_R2score_value}")
                DEBUG(f"test_rmse_value: {test_rmse_value}")
                # pdb.set_trace()

                INFO(f'year:{testyear} R2:{test_R2score_value} RMSE:{test_rmse_value}')

                # if(self.args.visualize):
                #     drawfigures(self.args, pred_best,test_y,'R2 :{}, RMSE : {}'.format(test_R2score_value, test_rmse_value),'Insta_MIR',testyear)


                # add_info=[str(testyear),'Instance',test_R2score_value,test_rmse_value,self.args.N,dimlength,topk,endday]
                # csvFile = open("InseasonBaseline_30m.csv", "a",newline='')
                # writer = csv.writer(csvFile)
                # writer.writerow(add_info)
                # csvFile.close()

                # pdb.set_trace()


if __name__ == '__main__':
    logging.getLogger().setLevel(20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default="./dataset/Without05NPY/", help="The input .npy file directory")
    parser.add_argument('--output_root', type=str, default="./output/result/MIRBase/", help="The output result directory")
    parser.add_argument('--model', type=str, default="mlp", help="The model to use: mlp, lr, ridge, randomforest")
    parser.add_argument('--N', type=int, default=100, help="The number of instances per bag")
    parser.add_argument('--visualize', action='store_true', help="visualize the attention map")
    parser.add_argument('--testyears', nargs='+', default='', help='2018 2019 2020 2021 2022')
    parser.add_argument('--enddays', nargs='+', default='', help='134 150 166 182 198 214 230 246 262 278')
    parser.add_argument('--usevars', nargs='+', default='', help='The variables to use: NDWI GCI EVI LSTday LSTnight ppt tmax tmean tmin vpdmax vpdmin awc cec som historical year All')
    args = parser.parse_args()
    args.testyears = [int(i) for i in args.testyears]
    args.enddays = [int(i) for i in args.enddays]

    INFO(f'args:-----------------------------')
    for k,v in vars(args).items():
        INFO(f"{k}: {v}")
    INFO(f'----------------------------------')

    mirbase = MIRBASE(args)
    mirbase.run_loop_enddays()


    

                    