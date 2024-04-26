import sys
import pdb
import os
from pathlib import Path
import csv
import math
from tqdm import tqdm
import argparse
import logging 
from logging import info as DEBUG
from logging import warning as INFO
from logging import error as ERROR
logging.basicConfig(level=30,format='[log: %(filename)s line:%(lineno)d] %(message)s')
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., pred=True):
        super().__init__()
        #out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.q = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.pred = pred
        if pred==True:
            self.fc2 = nn.Linear(hidden_features,1)
        else:
            self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x0 = x
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).squeeze(2)
        x += x0
        x1 = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.pred==False:
            x += x1
        x = x.squeeze(0)
        return x
    

class SVR(nn.Module):
    def __init__(self, N,dimlength, drop=0.):
        super().__init__()
        self.seq1 = nn.Sequential()
        ####(-1,13,211)
        self.seq1.add_module('f_conv1_1', nn.Linear(32,16))
        self.seq1.add_module('f_relu1_1', nn.ReLU(True))
        self.seq1.add_module('f_conv1_2', nn.Linear(16,8))
        self.seq1.add_module('f_relu1_2', nn.ReLU(True))
        self.seq2 = nn.Sequential()
        self.seq2.add_module('f_conv2_2', nn.Linear(8,1))
        # self.seq2.add_module('f_relu2_2', nn.ReLU(True))
    def forward(self, x):
        x=self.seq1(x)
        x=self.seq2(x)
        return x
    

class ConvTF(nn.Module):
    def __init__(self, N,dimlength, drop=0.):
        super().__init__()
        self.seq1 = nn.Sequential()
        ####(-1,13,211)
        self.seq1.add_module('f_conv1_1', nn.Linear(dimlength,128))
        self.seq1.add_module('f_bn1_1', nn.BatchNorm1d(128))
        self.seq1.add_module('f_relu1', nn.ReLU(True))
        self.seq2 = nn.Sequential()
        self.seq2.add_module('f_conv2', nn.Linear(128,64))
        self.seq2.add_module('f_bn2', nn.BatchNorm1d(64))
        self.seq2.add_module('f_relu2', nn.ReLU(True))
        ######(-1,4,211)
        self.Block1 = Mlp(in_features=64, hidden_features=32, act_layer=nn.GELU, drop=drop, pred=False)
        self.seq3 = nn.Sequential()
        self.seq3.add_module('f_fc3', nn.Linear(64, 32))
        self.seq3.add_module('f_bn3', nn.BatchNorm1d(32))
        self.seq3.add_module('f_relu3', nn.ReLU(True))
        self.seq4 = nn.Sequential()
        self.seq4.add_module('f_fc4', nn.Linear(32,1))
        # self.seq4.add_module('f_bn4', nn.BatchNorm1d(1))
        # self.seq4.add_module('f_relu4', nn.ReLU(True))
        # self.Block2 = Mlp(in_features=16, hidden_features=8, act_layer=nn.GELU, drop=drop, pred=True)

    def forward(self, x):
        # print(x.shape)
        x=self.seq1(x)
        x=self.seq2(x)
        # x=self.Block1(x)
        # print(x.shape)
        x=self.seq3(x)
        x=self.seq4(x)
        return x

class Encoder(nn.Module):
    def __init__(self, N,dimlength, drop=0.):
        super().__init__()
        self.seq1 = nn.Sequential()
        ####(-1,13,211)
        self.seq1.add_module('f_conv1_1', nn.Linear(dimlength,128))
        self.seq1.add_module('f_bn1_1', nn.BatchNorm1d(128))
        self.seq1.add_module('f_relu1', nn.ReLU(True))
        self.seq2 = nn.Sequential()
        self.seq2.add_module('f_conv2', nn.Linear(128,64))
        self.seq2.add_module('f_bn2', nn.BatchNorm1d(64))
        self.seq2.add_module('f_relu2', nn.ReLU(True))
        ######(-1,4,211)
        # self.Block1 = Mlp(in_features=64, hidden_features=32, act_layer=nn.GELU, drop=drop, pred=False)
        self.seq3 = nn.Sequential()
        self.seq3.add_module('f_fc3', nn.Linear(64, 32))
        self.seq3.add_module('f_bn3', nn.BatchNorm1d(32))
        self.seq3.add_module('f_relu3', nn.ReLU(True))
        # self.seq4 = nn.Sequential()
        # self.seq4.add_module('f_fc4', nn.Linear(32, 16))
        # self.seq4.add_module('f_bn4', nn.BatchNorm1d(16))
        # self.seq4.add_module('f_relu4', nn.ReLU(True))
        self.mu=nn.Linear(32,32)
        self.log_sigma=nn.Linear(32,32)
    def forward(self, x):
        # print(x.shape)
        x=self.seq1(x)
        # print(x.shape)

        x=self.seq2(x)
        # x=self.Block1(x)
        h=self.seq3(x)
        # h=self.seq4(x)
        mu=self.mu(h)
        log_sigma=self.log_sigma(h)
        sigma=torch.exp(log_sigma)
        return h, mu, sigma


class Decoder(nn.Module):
    def __init__(self, N,dimlength, drop=0.):
        super().__init__()
        self.seq1 = nn.Sequential()
        ####(-1,64)
        self.seq1.add_module('f_conv1_1', nn.Linear(32,32))
        self.seq1.add_module('f_bn1_1', nn.BatchNorm1d(32))
        self.seq1.add_module('f_relu1', nn.ReLU(True))
        self.seq2 = nn.Sequential()
        self.seq2.add_module('f_conv2', nn.Linear(32,64))
        self.seq2.add_module('f_bn2', nn.BatchNorm1d(64))
        self.seq2.add_module('f_relu2', nn.ReLU(True))
        self.seq3 = nn.Sequential()
        self.seq3.add_module('f_fc3', nn.Linear(64, 128))
        self.seq3.add_module('f_bn3', nn.BatchNorm1d(128))
        self.seq3.add_module('f_relu3', nn.ReLU(True))
        ######(-1,4,211)
        self.Block1 = Mlp(in_features=128, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        self.seq4 = nn.Sequential()
        self.seq4.add_module('f_fc4', nn.Linear(128, dimlength))
   
        # self.seq5 = nn.Sequential()
        # self.seq5.add_module('f_fc5', nn.Linear(math.ceil(N*dimlength/512)*512, N*dimlength))
        # self.seq5.add_module('f_bn5', nn.BatchNorm1d(N*dimlength))
    def forward(self, x):
        # print(x.shape)
        x=self.seq1(x)
        x=self.seq2(x)
        x=self.seq3(x)
        # x=self.Block1(x)
        x=self.seq4(x)
        # x=self.seq5(x)
        return x


def test(args, dataloader,pathstate):
    model_root = 'models'

    """ test """

    if pathstate=='best':
        my_net = torch.load(os.path.join(model_root, 'model_epoch_best.pth'))
    else:
        my_net = torch.load(os.path.join(model_root, 'model_epoch_current.pth'))
    my_net = my_net.eval()

    my_net = my_net.to(args.device)

    len_dataloader = len(dataloader)
    all_pre=[]
    all_label=[]
    for i,(X, y) in enumerate(dataloader):
        X=X.to(args.device)
        pred=my_net(X)
        y=y.to(args.device)
        # print(pred.shape)
        # print(y.shape)
        batch_size = len(y)
        pred=pred.view(batch_size,-1)
        y=y.view(batch_size,-1)
        all_pre.append(pred)
        all_label.append(y)
    all_pre=torch.cat(all_pre)
    all_label=torch.cat(all_label)
    all_pre=all_pre.to(args.device)
    all_label=all_label.to(args.device)   

    all_pred=all_pre.cpu().detach().numpy()
    all_labels=all_label.cpu().detach().numpy()            
    rmse=torchmetrics.MeanSquaredError(squared=False).to(args.device)
    rmse_value=rmse(torch.tensor(all_pred).to(args.device),torch.tensor(all_labels).to(args.device))
    R2score=torchmetrics.R2Score().to(args.device)
    R2score_value=R2score(torch.tensor(all_pred).to(args.device),torch.tensor(all_labels).to(args.device))
    np.save('VAEMIR_pred.npy',all_pred)
    np.save('VAEMIR_label.npy',all_labels)
   
    return R2score_value,rmse_value


def drawfigures(prepath,labelpath,outstring,name,testyear):
    x=np.load(labelpath)
    y=np.load(prepath)
    plt.ylabel('predicted')
    plt.xlabel('observed')
    plt.text(min(min(x),min(y)),max(max(x),max(y)),outstring)
    plt.scatter(x, y)
    plt.plot([min(min(x),min(y)),max(max(x),max(y))],[min(min(x),min(y)),max(max(x),max(y))], 'k--')
    plt.title(name)
    plt.savefig('Figures/'+str(testyear)+name+'.pdf',bbox_inches = 'tight',dpi=960)
    plt.close()
    return 0

def drawfigures(pre,label,outstring,name,testyear):
    x=label
    y=pre
    plt.ylabel('predicted')
    plt.xlabel('observed')
    plt.text(min(min(x),min(y)),max(max(x),max(y)),outstring)
    plt.scatter(x, y)
    plt.plot([min(min(x),min(y)),max(max(x),max(y))],[min(min(x),min(y)),max(max(x),max(y))], 'k--')
    plt.title(name)
    plt.savefig('Figures/'+str(testyear)+name+'.pdf',bbox_inches = 'tight',dpi=960)
    plt.close()

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, Xset ,yset,N,dimlength):
        self.Xset = Xset
        self.yset = yset
        self.N=N
        self.dimlength=dimlength
    
    def __len__(self):
        return len(self.Xset)
    
    def __getitem__(self, idx):
        # np.reshape(final_data,(final_data.shape[0],N,228)
        # X=torch.tensor(np.reshape(self.Xset[idx],(self.N,self.dimlength))).float()
        X=torch.tensor(self.Xset[idx]).float()
        y=torch.tensor(self.yset[idx]).float()
        return X, y

class AEDataSet(torch.utils.data.Dataset):
    def __init__(self, Xset ,N,dimlength):
        self.Xset = Xset
  
        self.N=N
        self.dimlength=dimlength
    
    def __len__(self):
        return len(self.Xset)
    
    def __getitem__(self, idx):
        # np.reshape(final_data,(final_data.shape[0],N,228)
        # X=torch.tensor(np.reshape(self.Xset[idx],(self.N,self.dimlength))).float()
        X=torch.tensor(self.Xset[idx]).float()
        return X
def concatdata(startyear,endyear,loaddir,usevar,endday):
# startyear=2008
# endyear=2017
# loaddir='dataset/NPY/'
    X=[]
    y=[]
    for year in list(range(startyear,endyear+1)):
        X.append(np.load(loaddir+'X_'+str(usevar)+'_'+str(year)+'_endday_'+str(endday)+'.npy',allow_pickle=True))
        y.append(np.load(loaddir+'y_'+str(usevar)+'_'+str(year)+'_endday_'+str(endday)+'.npy',allow_pickle=True))
    Xset=np.concatenate(X)
    yset=np.concatenate(y)
    return Xset, yset
def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
def isplit_by_n(ls, n,topk):
    for i in tqdm(range(0, len(ls), n)):
        cur=np.array(ls[i:i+n])
        orderlist=np.argsort(cur,axis=-1)
        yield orderlist[0:topk]

def split_by_n(ls, n,topk):
    return list(isplit_by_n(ls, n,topk))

def GetProtoList(dist,newX,n=100,topk=10):
    # composelist=split_by_n(dist.tolist(),n,topk)
    # DEBUG(f"composelist: {len(composelist)} {composelist[0].shape}")
    composelist = []
    dist_list = dist.tolist()
    for i in range(0, len(dist_list), n):
        cur = np.array(dist_list[i:i+n])
        orderlist = np.argsort(cur, axis=-1)
        composelist.append(orderlist[0:topk])
    DEBUG(f"composelist: {len(composelist)} {composelist[0].shape}")
    # pdb.set_trace()

    protolist=[]
    for i in range(len(composelist)):
        # proto=np.array([newX[i][k] for k in composelist[i]]).mean(axis=0)
        # DEBUG(f"proto: {proto.shape}")

        proto = []
        for k in composelist[i]:
            proto.append(newX[i][k])
        proto = np.array(proto)
        # DEBUG(f"proto: {proto.shape}")
        proto = proto.mean(axis=0)
        # DEBUG(f"proto: {proto.shape}")
        proto=proto.reshape(1,proto.shape[0])
        # DEBUG(f"proto: {proto.shape}")
        protolist.append(proto)
        # pdb.set_trace()
    protoarray=np.concatenate(protolist)
    # pdb.set_trace()
    return protoarray

def outputcounty(loaddir='dataset/NPY/',prepath='VAEMIR_pred.npy',labelpath='VAEMIR_label.npy',testyear=2016,usevar='ALL'):
    fipsfile=np.load(loaddir+'fips_'+str(usevar)+'_'+str(testyear)+'.npy')
    pre=np.load('VAEMIR_pred.npy')
    label=np.load('VAEMIR_label.npy')
    newdict={'FIPS':fipsfile.reshape(-1),'Prediction':pre.reshape(-1),'Yield':label.reshape(-1)}
    df=pd.DataFrame(newdict)

    df.to_excel('CountyResults_'+str(testyear)+'.xlsx',index=None)
    del df 
    return 0

def R2andRMSE(all_pre,all_label):
    # print(all_pre.shape)
    # print(all_label.shape)
    all_label=np.reshape(all_label,-1)
    all_pre=np.reshape(all_pre,-1)
    all_pre=torch.tensor(all_pre)
    all_label=torch.tensor(all_label)
    
    rmse=torchmetrics.MeanSquaredError(squared=False)
    rmse_value=rmse(all_pre,all_label)
    R2score=torchmetrics.R2Score()
    R2score_value=R2score(all_pre,all_label)
    np.save('VAEMIR_pred.npy',all_pre)
    np.save('VAEMIR_label.npy',all_label)
    return R2score_value.cpu().data.item(),rmse_value.cpu().data.item()


class VAEMIR():
    def __init__(self, args) -> None:
        self.args = args
    
    def train_vae(self, N, dimlength, train_loaderAE, model_root):
        Enc_patch = Encoder(N=N,dimlength=dimlength).to(self.args.device)
        Dec_patch = Decoder(N=N,dimlength=dimlength).to(self.args.device)

        un_optim_enc = torch.optim.Adam(Enc_patch.parameters(), lr=1e-3, weight_decay=0.0005)
        un_optim_dec=torch.optim.Adam(Dec_patch.parameters(), lr=1e-3, weight_decay=0.0005)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(self.args.epoch):
            rl=0
            epochloss=0
            INFO(f"Epoch {epoch}")
            for i, (data) in enumerate(tqdm(train_loaderAE)):
                #########################reconstruction phase#################
                data=data.to(self.args.device).float()
                Enc_patch.train()
                Dec_patch.train()
                un_optim_dec.zero_grad()
                un_optim_enc.zero_grad()
                H,mu, sigma =Enc_patch(data)
                std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
                code=mu + sigma * std_z.clone().detach().requires_grad_(False).to(self.args.device)
                recon=Dec_patch(code)
                ll = latent_loss(mu, sigma)
                recon_loss=criterion(data,recon)+ll
                loss=recon_loss
                loss.backward()
                un_optim_dec.step()
                un_optim_enc.step()
                epochloss+=loss.item()
            INFO(f"epochloss: {epochloss/len(train_loaderAE)}")
        torch.save(Enc_patch, model_root+"/Encoder_best.pth")
        torch.save(Dec_patch, model_root+"/Decoder_best.pth")

    def run(self):
        model_root = self.args.model_dir+'/models/'
        # testyears=[2020]
        # endday=285
        topks=[100]
        # topks=[1,10,20,30,40,50,60,65,70,80,90,100]###### top k values to compute proto#########


        #####usevar can be ['Satellite','GLDAS','LST','PRISM','All']
        usevar='All'
        # for endday in range(182, 280, 16):
        for endday in [278]:
            for testyear in self.args.testyears:
                INFO(f"testyear: {testyear}")
                train_x,train_y=concatdata(2008,testyear-1,self.args.input_root,usevar,endday)
                DEBUG(f"train_x.shape: {train_x.shape}")
                DEBUG(f"train_y.shape: {train_y.shape}")
                dimlength=train_x.shape[1]//self.args.N
                DEBUG(f"dimlength: {dimlength}")
                test_x,test_y=concatdata(testyear,testyear,self.args.input_root,usevar,endday)
                DEBUG(f"test_x.shape: {test_x.shape}")
                DEBUG(f"test_y.shape: {test_y.shape}")
                scaler1 = preprocessing.MinMaxScaler()
                XtrainAE=np.reshape(train_x,(-1,dimlength))###(bs*N,dimlength)
                XtestAE=np.reshape(test_x,(-1,dimlength))###(bs*N,dimlength)
                DEBUG(f"XtrainAE.shape: {XtrainAE.shape}")
                DEBUG(f"XtestAE.shape: {XtestAE.shape}")
                X_AE=np.concatenate((XtrainAE,XtestAE),axis=0)
                X_AE=scaler1.fit_transform(X_AE)
                ytrain = train_y
                ytest = test_y
                traindataAE=AEDataSet(X_AE,self.args.N,dimlength)
                train_loaderAE = torch.utils.data.DataLoader(traindataAE, batch_size=self.args.batchsize, shuffle=True, num_workers=0)
                # pdb.set_trace()
                #############Unsupervised VAE Training ########################
                if (self.args.TrainVAE==True):
                    self.train_vae(self.args.N, dimlength, train_loaderAE, model_root)
                ####################After VAE training, compute prototype list####################

                Enc_patch=torch.load(model_root+"/Encoder_best.pth")
                Dec_patch=torch.load(model_root+"/Decoder_best.pth")

                Test_loaderAE = torch.utils.data.DataLoader(traindataAE, batch_size=self.args.batchsize, shuffle=False, num_workers=0)
                all_data=[]
                all_recon=[]
                for i, (data) in enumerate(Test_loaderAE):
                    #########################reconstruction phase#################
                    # DEBUG(f"i: {i} data.shape: {data.shape}")
                    data=data.to(self.args.device).float()
                    Enc_patch.eval()
                    Dec_patch.eval()
                    H,mu, sigma =Enc_patch(data)
                    std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
                    code=mu + sigma * std_z.clone().detach().requires_grad_(False).to(self.args.device)
                    recon=Dec_patch(code)
                    # DEBUG(f"recon.shape: {recon.shape}")
                    all_data.append(data)
                    all_recon.append(recon)
                    # pdb.set_trace()
                
                all_data=torch.cat(all_data).to(self.args.device)
                all_recon=torch.cat(all_recon).to(self.args.device)
                DEBUG(f"all_data.shape: {all_data.shape}")
                DEBUG(f"all_recon.shape: {all_recon.shape}")
                dist = (all_data - all_recon).pow(2).sum(1).sqrt().cpu().detach().numpy()
                DEBUG(f"dist.shape: {dist.shape}")
                # pdb.set_trace()
                ###########compute proto index and get protolist###########
                for topk in topks:
                    INFO(f"year: {testyear} endday: {endday} topk: {topk}")
                    output_dir = self.args.output_root + str(testyear) + "/" + str(endday) + "/" + str(topk) + "/"
                    Path(output_dir).mkdir(parents=True, exist_ok=True)

                    newX=np.concatenate((train_x,test_x),axis=0)
                    newX=newX.reshape(newX.shape[0],self.args.N,-1)
                    DEBUG(f"newX.shape: {newX.shape}")
                    protoX=GetProtoList(dist,n=self.args.N,topk=topk,newX=newX)
                    DEBUG(f"protoX.shape: {protoX.shape}")
                    protoX=scaler1.fit_transform(protoX)
                    XtrainRegre=protoX[0:train_x.shape[0]]
                    XtestRegre=protoX[train_x.shape[0]:]
                    DEBUG(f"XtrainRegre.shape: {XtrainRegre.shape}")
                    DEBUG(f"XtestRegre.shape: {XtestRegre.shape}")
                    traindata=MyDataSet(XtrainRegre,ytrain,self.args.N,dimlength)
                    DEBUG(f"traindata: {len(traindata)}")
                    DEBUG(f"{traindata[0][0].shape}")
                    DEBUG(f"{traindata[0][1].shape}")
                    # pdb.set_trace()
                    #####################val##########################
                    trainsize=int(len(traindata)*0.9)
                    valsize=len(traindata)-trainsize
                    trainset,valset=torch.utils.data.random_split(traindata,[trainsize,valsize])

                    testdata=MyDataSet(XtestRegre,ytest,self.args.N,dimlength)

                    train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batchsize, shuffle=True, num_workers=0)

                    val_loader = torch.utils.data.DataLoader(valset, batch_size=self.args.batchsize, shuffle=True, num_workers=0)
                    ###############################################val#################
                    test_loader = torch.utils.data.DataLoader(testdata, batch_size=self.args.batchsize, shuffle=False, num_workers=0)
                    torch.cuda.empty_cache()
                    ##############Train regressor using Prototype###########################
                    ###################################SKlearn Module############################################
                    train_x2=XtrainRegre
                    train_y2=ytrain.reshape(-1)
                    test_x2=XtestRegre
                    test_y2=ytest.reshape(-1)
                    np.save(output_dir+"/train_x.npy",train_x2)
                    np.save(output_dir+"/train_y.npy",train_y2)
                    rbf_svr = MLPRegressor(hidden_layer_sizes=(128,64), learning_rate_init=0.001, max_iter=300, alpha=3)
                    rbf_svr.fit(train_x2, train_y2)
                    train_pred = rbf_svr.predict(train_x2)
                    test_pred = rbf_svr.predict(test_x2)
                    
                    test_pred=(test_pred.reshape(-1,1))
                    test_y=(test_y2.reshape(-1,1))
                    np.save(output_dir+"/test_pred.npy",test_pred)
                    np.save(output_dir+"/final_y.npy",test_y)
                    test_R2score_value,test_rmse_value=R2andRMSE(test_pred,test_y2)
                    
                    # drawfigures(test_pred,test_y2,'R2 :{}, RMSE : {}'.format(test_R2score_value, test_rmse_value),'VAEMIR',testyear)
                    INFO(f"year: {testyear} R2: {test_R2score_value} RMSE: {test_rmse_value} topk: {topk}")

                    # outputcounty(loaddir=self.args.input_root,prepath='VAEMIR_pred.npy',labelpath='VAEMIR_label.npy',testyear=testyear,usevar=usevar)
                    # add_info=[str(testyear),'VAEMIR',test_R2score_value,test_rmse_value,self.args.N,dimlength,topk,endday]
                    # csvFile = open("InseasonVAEMIR_30m.csv", "a",newline='')
                    # writer = csv.writer(csvFile)
                    # writer.writerow(add_info)
                    # csvFile.close()
                    ###################################SKlearn Module############################################

if __name__ == '__main__':
    logging.getLogger().setLevel(30)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default="")
    parser.add_argument('--output_root', type=str, default="")
    parser.add_argument('--model_dir', type=str, default="")
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--TrainVAE', action="store_true")
    parser.add_argument('--batchsize', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--testyears', nargs='+', default='', help='')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INFO(f"device: {args.device}")
    args.testyears = [int(i) for i in args.testyears]
    vaemir = VAEMIR(args)
    vaemir.run()

    
