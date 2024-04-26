import pdb
from tqdm import tqdm
from pathlib import Path
import os
import sys
import argparse
import logging 
from logging import debug as DEBUG
from logging import info as INFO
from logging import error as ERROR
logging.basicConfig(level=0,format='[log: %(filename)s line:%(lineno)d] %(message)s')
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import glob


def AggregatedMean(prefix,num_list,pixels):
    # DEBUG(f'prefix: {prefix}')
    # DEBUG(f'pixels: {pixels.shape}')
    for day in num_list:
        # DEBUG(f'day: {day}')
        aggrelist=np.arange(day-8,day+8,1)
        # DEBUG(f'aggrelist: {aggrelist}')
        aggrecolumns=list(map(lambda x:prefix+str(x), aggrelist))
        # DEBUG(f'aggrecolumns: {aggrecolumns}')
        filtered_columns=[x for x in aggrecolumns if x in pixels.columns]
        # DEBUG(f'filtered_columns: {filtered_columns}')
        curdf=pixels[filtered_columns]
        # DEBUG(f'curdf: {curdf.shape}')
        meandf=curdf.replace(0,np.nan).mean(axis=1)
        # DEBUG(f'meandf: {meandf.shape}')
        pixels[prefix+'aggre_'+str(day)]=meandf
        # DEBUG(f'pixels: {pixels.shape}')
        # pdb.set_trace()
    newcol_list=list(map(lambda x:prefix+'aggre_'+str(x), num_list))
    # DEBUG(f'newcol_list: {newcol_list}')
    for i,column in enumerate(newcol_list):
        pixels[column].fillna(pixels[newcol_list].mean(axis=1),inplace=True)##use mean of all time to fill nan
    # pdb.set_trace()
    return pixels,newcol_list

class Process():
    def __init__(self, args) -> None:
        self.args=args

    def post_process(self):
        historical=pd.read_csv(args.output_root+'/ave_yield.csv')
        INFO(f"historical: {type(historical)} {historical.shape}")
        INFO(f'columns: {historical.columns}')
        # pdb.set_trace()
        corn=pd.read_csv(args.output_root+'/yield.csv')
        INFO(f"corn: {type(corn)} {corn.shape}")
        INFO(f"columns: {corn.columns}")

        corn=corn.fillna(value=0)
        INFO(f'corn: {corn.shape}')
        INFO(f'corn type: {type(corn["FIPS"])} {type(corn["FIPS"][0])} {corn["FIPS"].dtypes}')
        corn['FIPS']=corn['FIPS'].map(str)
        INFO(f'corn: {corn.shape}')
        INFO(f'corn type: {type(corn["FIPS"])} {type(corn["FIPS"][0])} {corn["FIPS"].dtypes}')
        
        historical=historical.fillna(value=0)
        INFO(f'historical: {historical.shape}')
        INFO(f'historical type: {type(historical["FIPS"])} {type(historical["FIPS"][0])} {historical["FIPS"].dtypes}')
        historical['FIPS']=historical['FIPS'].map(str)
        INFO(f'historical: {historical.shape}')
        INFO(f'historical type: {type(historical["FIPS"])} {type(historical["FIPS"][0])} {historical["FIPS"].dtypes}')
        # pdb.set_trace()
        bag_num=0
        oritotallen=0
        cleanedtotallen=0
        raw_bagnum=0
        #####usevar can be ['Satellite','GLDAS','LST','PRISM','All']
        # pdb.set_trace()
        for year in tqdm(range(self.args.startyear,self.args.endyear+1)):
            INFO(f'year: {year}')
            bags=[]
            labels=[]
            FIPSdata=[]
            location_list=[]
            confidence_list=[]
            for state in tqdm(self.args.statefp):
                filelist=glob.glob(self.args.data_dir+'Data_'+str(year)+'_STATE_'+str(state).rjust(2,'0')+'*.csv')
                INFO(f"state: {state} filelist: {len(filelist)}")
                DEBUG(f"{filelist[:3]}")
                # pdb.set_trace()
                for datapath in filelist:
                    DEBUG(f"datapath: {datapath}")
                    # pdb.set_trace()
                    if os.path.getsize(datapath)<10:
                        continue
                    try:
                        pixels=pd.read_csv(datapath)
                    except:
                        continue
                    DEBUG(f'pixels: {type(pixels)} {pixels.shape}')
                    DEBUG(f'{pixels.iloc[:3,:3]}')
                    if(pixels.shape[0]<self.args.N):
                        continue
                    oritotallen+=len(pixels.index)
                    pixels['FIPS']=str(state)+datapath.split('_')[-1].strip('.csv')
                    num_list=list(range(self.args.startday,self.args.endday+1,16))
                    DEBUG(f'num_list: {num_list}')
                    pixels=pixels.loc[(pixels.iloc[:,7:371]!=0).any(axis=1)]###drop zero rows
                    DEBUG(f'pixels: {pixels.shape}')
                    # pdb.set_trace()
                    if pixels.empty:
                        continue
                    cur=historical[historical['year']==year]
                    DEBUG(f'cur: {cur.shape}')
                    fixyearfips=cur.loc[cur['FIPS']==pixels['FIPS'].iloc[0]]
                    DEBUG(f'fixyearfips: {fixyearfips.shape}')
                    if fixyearfips.empty:
                        continue
                    pixels['historical yield']=fixyearfips['average_yield'].values[0]
                    pixels['year']=year
                    DEBUG(f'pixels: {pixels.shape}')
                    # # pdb.set_trace()

                    LoadList=[]
                    ####ALL daily variables ##########
                        # ['NDVI','NDWI','GCI','EVI','Evap_tavg','LST_Day_1km','LST_Night_1km',\
                    # 'PotEvap_tavg','RootMoist_inst',\
                        #'ppt','tdmean','tmax','tmean','tmin','vpdmax','vpdmin']
                    #####usevar can be ['Satellite','GLDAS','LST','PRISM','All']
                    if self.args.usevar=='All':
                        var_list=['NDWI','GCI','EVI',\
                        'LST_Day_1km','LST_Night_1km',\
                        'ppt','tmax','tmean','tmin','vpdmax','vpdmin']
                        for var in var_list:
                            pixels,cur_list=AggregatedMean(var+'_',num_list,pixels)
                            # DEBUG(f'pixels: {pixels.shape}')
                            # DEBUG(f'cur_list: {cur_list}')
                            # pdb.set_trace()
                            LoadList+=cur_list
                        ###########Not daily variables##############
                        ######'awc','cec','som'
                        pixels=pixels[LoadList].join(pixels.loc[:,['FIPS','awc','cec','som','historical yield','year','.geo', 'confidence']])
                        DEBUG(f'pixels: {pixels.shape}')
                        DEBUG(f'columns: {pixels.columns}')
                        DEBUG(f'{pixels[".geo"][0]}')
                        # pdb.set_trace()
                        # pixels=pd.concat(pixels.reset_index(drop=True),pd.DataFrame(np.repeat()))
                    elif self.args.usevar=='Satellite':
                        var_list=['NDVI','NDWI','GCI','EVI']
                        for var in var_list:
                            pixels,cur_list=AggregatedMean(prefix=var+'_',num_list=num_list,pixels=pixels)
                            LoadList+=cur_list
                        ###########Not daily variables##############
                        ######'awc','cec','som'
                        pixels=pixels[LoadList].join(pixels.loc[:,['FIPS']])
                    elif self.args.usevar=='GLDAS':
                        var_list=['Evap_tavg','PotEvap_tavg','RootMoist_inst']
                        for var in var_list:
                            pixels,cur_list=AggregatedMean(prefix=var+'_',num_list=num_list,pixels=pixels)
                            LoadList+=cur_list
                        ###########Not daily variables##############
                        ######'awc','cec','som'
                        pixels=pixels[LoadList].join(pixels.loc[:,['FIPS']])
                    elif self.args.usevar=='LST':
                        var_list=['LST_Day_1km','LST_Night_1km']
                        for var in var_list:
                            pixels,cur_list=AggregatedMean(prefix=var+'_',num_list=num_list,pixels=pixels)
                            LoadList+=cur_list
                        ###########Not daily variables##############
                        ######'awc','cec','som'
                        pixels=pixels[LoadList].join(pixels.loc[:,['FIPS']])                
                    elif self.args.usevar=='PRISM':
                        var_list=['ppt','tdmean','tmax','tmean','tmin','vpdmax','vpdmin']
                        for var in var_list:
                            pixels,cur_list=AggregatedMean(prefix=var+'_',num_list=num_list,pixels=pixels)
                            LoadList+=cur_list
                        ###########Not daily variables##############
                        ######'awc','cec','som'
                        pixels=pixels[LoadList].join(pixels.loc[:,['FIPS']])
                    else:
                        raise ValueError('wrong usevar')
                    
                    pixels=pixels.dropna()
                    DEBUG(f'pixels: {pixels.shape}')
                    DEBUG(f'columns: {pixels.columns}')
                    for i in range(11):
                        DEBUG(f'columns: {pixels.columns.tolist()[14*i:14*(i+1)]}')
                    DEBUG(f'columns: {pixels.columns.tolist()[-7:]}')
                    cleanedtotallen+=len(pixels.index)

                    # pixels.to_csv(self.args.output_root+'/Cleaned_'+self.args.usevar+'_'+datapath.split('/')[-1],index=False)
                    # pdb.set_trace()

                    curcorn=corn.loc[corn['year']==int(year)]
                    DEBUG(f'FIPS: {pixels["FIPS"].unique()}')
                    if(pixels['FIPS'].unique().shape[0]!=1):
                        raise ValueError('FIPS not unique')
                    fips=pixels['FIPS'][0]
                    DEBUG(f'fips: {fips}')
                    # pdb.set_trace()
                    if curcorn[curcorn['FIPS']==fips]['yield'].any():###如果该位置有yield值，则记下来
                        yield_value=curcorn[curcorn['FIPS']==fips]['yield'].values[0]
                        DEBUG(f'yield_value: {yield_value}')
                        DEBUG(f'pixels: {len(pixels)}')
                        if len(pixels)<self.args.N:
                            continue
                        instance=pixels.sample(self.args.N)###100 instance with 46 features
                        DEBUG(f'instance: {instance.shape}')
                        location=instance['.geo'].to_numpy()
                        DEBUG(f'location: {location.shape}')
                        confidence=instance['confidence'].to_numpy()
                        DEBUG(f'confidence: {confidence.shape}')
                        instance=instance.drop(['FIPS', '.geo', 'confidence'],axis=1)
                        DEBUG(f'instance: {instance.shape}')
                        for i in range(11):
                            DEBUG(f'columns: {instance.columns.tolist()[14*i:14*(i+1)]}')
                        DEBUG(f'columns: {instance.columns.tolist()[-5:]}')
                        for i in range(11):
                            DEBUG(f'check csv: {instance.iloc[1,14*i]}')
                        DEBUG(f'check csv: {instance.iloc[1,-5:]}')
                        
                        instance=instance.to_numpy()
                        DEBUG(f'instance: {instance.shape}')
                        for i in range(11):
                            DEBUG(f'check npy: {instance[1,14*i]}')
                        DEBUG(f'check npy: {instance[1,-5:]}')
                        instance=np.reshape(instance,-1)
                        DEBUG(f'instance: {instance.shape}')
                        for i in range(11):
                            DEBUG(f'check stack npy: {instance[159+14*i]}')
                        DEBUG(f'check stack npy: {instance[159+159-5:159+159]}')
                        label=yield_value/14.87 ####convert to t/ha
                        DEBUG(f'label: {label}')
                        DEBUG(f'fips: {fips}')
                        bags.append(instance)###1 bag 
                        labels.append(label)
                        FIPSdata.append(fips)
                        location_list.append(location)
                        confidence_list.append(confidence)
                        bag_num+=1
                        # pdb.set_trace()
                    del pixels
                DEBUG(f'bags: {len(bags)}')
            
            DEBUG(f'bags: {len(bags)} {bags[0].shape}')
            DEBUG(f'location_list: {len(location_list)} {location_list[0].shape}')
            bags=np.array(bags)### N*100*46  all bags
            DEBUG(f'bags: {bags.shape}')
            location_list=np.array(location_list)
            DEBUG(f'location_list: {location_list.shape}')
            confidence_list=np.array(confidence_list)
            DEBUG(f'confidence_list: {confidence_list.shape}')
            labels=np.array(labels)
            labels=labels.reshape(-1,1)   
            FIPSdata=np.array(FIPSdata).reshape(-1,1)
            DEBUG(f'year:{year}, bags: {bags.shape}, labels:{labels.shape}, FIPS:{FIPSdata.shape} location_list:{location_list.shape} confidence_list:{confidence_list.shape}')
            # pdb.set_trace()
            output_dir = self.args.output_root+'/'+str(self.args.N)+'from1100/'
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            np.save(output_dir+'X_'+str(self.args.usevar)+'_'+str(year)+'_endday_'+str(self.args.endday)+'.npy',bags)
            np.save(output_dir+'y_'+str(self.args.usevar)+'_'+str(year)+'_endday_'+str(self.args.endday)+'.npy',labels)
            np.save(output_dir+'fips_'+str(self.args.usevar)+'_'+str(year)+'_endday_'+str(self.args.endday)+'.npy',FIPSdata)
            np.save(output_dir+'loc_'+str(self.args.usevar)+'_'+str(year)+'_endday_'+str(self.args.endday)+'.npy',location_list)
            np.save(output_dir+'conf_'+str(self.args.usevar)+'_'+str(year)+'_endday_'+str(self.args.endday)+'.npy',confidence_list)
        INFO(f'bag_num: {bag_num}')
        INFO(f'oritotallen: {oritotallen} cleanedtotallen: {cleanedtotallen}')


    def process_yield(self):
        input_path = self.args.input_root + '/yieldwithfips.csv'
        yield_data = pd.read_csv(input_path)
        INFO(f'yield_data shape: {yield_data.shape}\n{yield_data.columns}')
        yield_data = yield_data.rename(columns={'Year': 'year', 'Value': 'yield'})
        yield_data = yield_data[['year', 'State', 'County', 'yield', 'FIPS']]
        INFO(f'yield_data shape: {yield_data.shape}')
        yield_data = yield_data.rename(columns={'State': 'state_name', 'County': 'county_name'})
        INFO(f'yield_data shape: {yield_data.shape}\n{yield_data.columns}')
        yield_data = yield_data.dropna()
        # yield_data = yield_data[yield_data.year >= 2001]
        yield_data = yield_data.sort_values(by=['FIPS','year'])
        INFO(f'fips: {len(yield_data.FIPS.unique())}')
        # yield_data.to_csv(self.args.output_root+'/yield.csv', index=False)
        # pdb.set_trace()

        # for fips in tqdm(yield_data['FIPS'].unique()):
        #     exist = 1
        #     for year in range(2001, 2023, 1):
        #         if(fips, year) not in zip(yield_data['FIPS'], yield_data['year']):
        #             exist = 0
        #             break
        #     if(exist == 0):
        #         yield_data.drop(yield_data[yield_data['FIPS'] == fips].index, inplace=True)

        INFO(f'drop fips: {len(yield_data.FIPS.unique())}')
        yield_data.to_csv(self.args.output_root+'/yield.csv', index=False)

    def cal_ave_yield(self):
        input_path = self.args.output_root + '/yield.csv'
        yield_data = pd.read_csv(input_path)
        INFO(f'yield_data shape: {yield_data.shape}')
        INFO(f'yield_data columns: {yield_data.columns}')
        ave_yield_data = yield_data[['year', 'FIPS', 'yield']]
        INFO(f'ave_yield_data shape: {ave_yield_data.shape}')
        # ave_yield_data.to_csv(self.args.output_root+'/ave_yield.csv', index=False)
        # pdb.set_trace()
        for index, row in tqdm(ave_yield_data.iterrows(), total=ave_yield_data.shape[0]):
            DEBUG(f'index: {index}\nrow: {row}')
            year = int(row['year'])
            yield_data = row['yield']
            FIPS = int(row['FIPS'])
            DEBUG(f'year: {type(year)} {year}')
            DEBUG(f'yield: {type(yield_data)} {yield_data}')
            DEBUG(f'FIPS: {type(FIPS)} {FIPS}')
            if(year < 2008):
                continue
            sum = 0
            exist = 1
            for i in range(5):
                pre_year = year - i - 1
                DEBUG(f'pre_year: {pre_year}')
                DEBUG(f'FIPS: {FIPS}')
                y = ave_yield_data[(ave_yield_data.year == year-i-1) & (ave_yield_data.FIPS == FIPS)]['yield'].values
                DEBUG(f'pre_year: {pre_year}, FIPS: {FIPS}, yield: {y} {type(y)} {y.shape}')
                if(y.shape[0] == 0):
                    # pdb.set_trace()
                    exist = 0
                    break
                y = y[0]
                sum += y
            if(exist == 0):
                ave_yield_data.loc[index, 'average_yield'] = -1
                continue
            average_yield = sum / 5
            DEBUG(f'average_yield: {average_yield}')
            ave_yield_data.loc[index, 'average_yield'] = average_yield
            # pdb.set_trace()
        INFO(f'ave_yield_data shape: {ave_yield_data.shape}')
        
        # ave_yield_data = ave_yield_data.drop(['yield'])
        ave_yield_data = ave_yield_data[ave_yield_data.year >= 2008]
        INFO(f'ave_yield_data shape: {ave_yield_data.shape}')
        ave_yield_data = ave_yield_data.dropna()
        INFO(f'drop nan shape: {ave_yield_data.shape}')
        
        ave_yield_data = ave_yield_data.drop(columns=['yield'])
        INFO(f'ave_yield_data shape: {ave_yield_data.shape}')
        ave_yield_data = ave_yield_data.sort_values(by=['FIPS','year'])
        ave_yield_data = ave_yield_data[ave_yield_data['average_yield'] > 0]
        INFO(f'drop minus shape: {ave_yield_data.shape}')
        ave_yield_data.to_csv(self.args.output_root+'/ave_yield.csv', index=False)


    def process_vi(self):
        input_path = self.args.input_root + '/county_VI.csv'
        vi_data = pd.read_csv(input_path)
        INFO(f'vi_data shape: {vi_data.shape}')
        INFO(f'vi_data columns: {vi_data.columns}')
        NDWI_data = vi_data[['year', 'FIPS','NDVI_058', 'NDVI_074', 'NDVI_090', 'NDVI_106', 'NDVI_122',\
                                'NDVI_138', 'NDVI_154', 'NDVI_170', 'NDVI_186', 'NDVI_202', 'NDVI_218',\
                                'NDVI_234', 'NDVI_250', 'NDVI_266', 'NDVI_282', 'NDVI_298', 'NDVI_314','NDVI_330']]
        INFO(f'NDWI_data shape: {NDWI_data.shape}')
        NDWI_data.to_csv(self.args.output_root+'/NDVI_data.csv', index=False)
        vi_data.to_csv(self.args.output_root+'/VI_data.csv', index=False)


    def process_prism(self):
        vpd_input_path = self.args.input_root + '/county_PRISM_vpd.csv'
        vpd_data = pd.read_csv(vpd_input_path)
        INFO(f'vpd_data shape: {vpd_data.shape}')
        INFO(f'vpd_data columns: {vpd_data.columns}')

        ppt_input_path = self.args.input_root + '/county_PRISM_ppt.csv'
        ppt_data = pd.read_csv(ppt_input_path)
        INFO(f'ppt_data shape: {ppt_data.shape}')
        INFO(f'ppt_data columns: {ppt_data.columns}')

        prism_data = vpd_data.join(ppt_data.set_index(['year', 'FIPS']), on=['year', 'FIPS'])
        INFO(f'prism_data shape: {prism_data.shape}')
        INFO(f'prism_data columns: {prism_data.columns}')
        prism_data.to_csv(self.args.output_root+'/prism_data.csv', index=False)

    def process_gldas(self):
        input_path = self.args.input_root + '/county_GLDAS.csv'
        gldas_data = pd.read_csv(input_path)
        INFO(f'gldas_data shape: {gldas_data.shape}')
        INFO(f'gldas_data columns: {gldas_data.columns}')
        gldas_data.to_csv(self.args.output_root+'/gldas_data.csv', index=False)

    def process_soil(self):
        soil_types = ['cec', 'awc', 'som']
        soil_datas = []
        for soil_type in soil_types:
            input_path = self.args.input_root + '/county_'+soil_type+'.csv'
            soil_data = pd.read_csv(input_path)
            INFO(f'soil_data shape: {soil_data.shape}')
            INFO(f'soil_data columns: {soil_data.columns}')
            soil_data = soil_data.rename(columns={"soil": soil_type})
            INFO(f'soil_data columns: {soil_data.columns}')
            soil_datas.append(soil_data)

        for i in range(len(soil_types)):
            if(i == 0):
                soil_data = soil_datas[i]
            else:
                soil_data = soil_data.join(soil_datas[i].set_index(['year', 'FIPS']), on=['year', 'FIPS'])
        INFO(f'soil_data shape: {soil_data.shape}')
        INFO(f'soil_data columns: {soil_data.columns}')
        soil_data.to_csv(self.args.output_root+'/soil_data.csv', index=False)

    def check_data(self):
        root = '/mnt/d/data/1100pixels/data/'

        datalist = []
        statelist = {'38':[],'46':[],'27':[],'55':[],'19':[],'17':[],'18':[],'39':[],'29':[],'20':[],'31':[],'26':[]}

        for file in os.listdir(root):
            file = file.split('.')[0]
            DEBUG(f'file: {file}')
            datalist.append(file)
            year = file.split('_')[1]
            state = file.split('_')[3]
            county = file.split('_')[5]
            DEBUG(f'state: {state} county: {county} year: {year}')
            # pdb.set_trace()
            if(county not in statelist[state]):
                statelist[state].append(county)
        # pdb.set_trace()
        for state in tqdm(statelist.keys()):
            DEBUG(f'state: {state} counties: {len(statelist[state])}')
            for county in statelist[state]:
                for year in range(2008, 2023):
                    filename = 'Data_' + str(year) + '_STATE_' + state + '_COUNTY_' + county
                    if(filename not in datalist):
                        ERROR(f'filename: {filename} not in datalist')

    def check_zeyu(self):
        input_path = '/mnt/d/data/VAE_MIR_CODE/input/zeyu/corn_yield_US_1975_2021.csv'
        yield_data = pd.read_csv(input_path)
        INFO(f'yield_data shape: {yield_data.shape}\n{yield_data.columns}')
        yield_data = yield_data[['year', 'FIPS', 'yield']]
        yield_data = yield_data.sort_values(by=['FIPS','year'])
        INFO(f'fips: {len(yield_data["FIPS"].unique())}')
        for fips in tqdm(yield_data['FIPS'].unique()):
            exist = 1
            for year in range(2001, 2020, 1):
                if(fips, year) not in zip(yield_data['FIPS'], yield_data['year']):
                    exist = 0
                    break
            if(exist == 0):
                yield_data.drop(yield_data[yield_data['FIPS'] == fips].index, inplace=True)
        INFO(f'fips: {len(yield_data["FIPS"].unique())}')
        yield_data.to_csv(self.args.output_root+'/zeyudrop.csv', index=False)


    def check_empty(self):
        datapath = '/mnt/d/data/3000pixels/Combined_30m_2008_2022/Data_2022_STATE_46_COUNTY_033.csv'
        # data = pd.read_csv(datapath)
        try:
            data = pd.read_csv(datapath)
        except:
            pass
    
    def check_npy(self):
        for year in range(2008, 2023):
            for endday in [134, 150, 166, 182, 198, 214, 230, 246, 262, 278]:
                INFO(f'year: {year} endday: {endday}')
                x_path = self.args.output_root+'/'+self.args.processed_data_dir+'/X_All_' + str(year) + '_endday_'+str(endday)+'.npy'
                x = np.load(x_path, allow_pickle=True)
                INFO(f'data shape: {x.shape}')
                y_path = self.args.output_root+'/'+self.args.processed_data_dir+'/y_All_' + str(year) + '_endday_'+str(endday)+'.npy'
                y = np.load(y_path, allow_pickle=True)
                INFO(f'data shape: {y.shape}')
                fips_path = self.args.output_root+'/'+self.args.processed_data_dir+'/fips_All_' + str(year) + '_endday_'+str(endday)+'.npy'
                fips = np.load(fips_path, allow_pickle=True)
                INFO(f'data shape: {fips.shape}')
            pdb.set_trace()


    def cal_result(self):
        path='/mnt/d/data/VAE_MIR_CODE/output/1000_0521/randomforest/1.txt'
        with open(path, 'r') as f:
            lines = f.readlines()
            rmse_list = [{},{},{},{},{}]
            r2_list = [{},{},{},{},{}]
            endday = 0
            year = 0
            for i, line in enumerate(lines):
                DEBUG(f'line: {line}')
                if(len(line)<1):
                    continue
                if(line.split(' ')[0] == 'endday:'):
                    endday = int(line.split(' ')[1])
                    DEBUG(f'endday: {endday}')
                    year = int(line.split(' ')[5])
                    DEBUG(f'year: {year}')
                else:
                    r2 = float(line.split(' ')[1].split(':')[1])
                    DEBUG(f'r2: {r2}')
                    rmse = float(line.split(' ')[2].split(':')[1])
                    DEBUG(f'rmse: {rmse}')
                    rmse_list[year-2018][endday] = rmse
                    r2_list[year-2018][endday] = r2
                    # pdb.set_trace()

            # for i, line in enumerate(lines):
            #     if(len(line)<1):
            #         continue
            #     if(line.split(' ')[0] == 'endday:'):
            #         endday = int(line.split(' ')[1])
            #         DEBUG(f'endday: {endday}')
            #         year = int(line.split(' ')[5])
            #         DEBUG(f'year: {year}')
            #     elif(line.split(' ')[0] == 'best'):
            #         r2 = float(line.split(' ')[3])
            #         DEBUG(f'r2: {r2}')
            #         rmse = float(line.split(' ')[5])
            #         DEBUG(f'rmse: {rmse}')
            #         rmse_list[year-2018][endday] = rmse
            #         r2_list[year-2018][endday] = r2
            #         # pdb.set_trace()

        rmse_ave = {}
        r2_ave = {}
        for i in range(5):
            DEBUG(f'rmse: {rmse_list[i]}')
            DEBUG(f'r2: {r2_list[i]}')
            if(i == 0):
                for key in rmse_list[i].keys():
                    rmse_ave[key] = rmse_list[i][key]
                    r2_ave[key] = r2_list[i][key]
                continue
            for key in rmse_list[i].keys():
                rmse_ave[key] += rmse_list[i][key]
                r2_ave[key] += r2_list[i][key]
        for key in rmse_ave.keys():
            rmse_ave[key] /= 5
            r2_ave[key] /= 5
        DEBUG(f'rmse_ave: {rmse_ave}')
        DEBUG(f'r2_ave: {r2_ave}')
        
    def run(self):
        if(self.args.process == 'pre'):
            self.process_yield()

            self.cal_ave_yield()

            # self.process_vi()

            # self.process_prism()

            # self.process_gldas()

            # self.process_soil()

            

        elif(self.args.process == 'post'):
            self.post_process()
        elif(self.args.process == 'check'):
            self.check()
        else:
            raise('process error')


    def check(self):
        # self.check_npy()

        self.check_data()

        # self.check_zeyu()

        # self.check_empty()

        # self.check_npy()

        # self.cal_result()




if __name__ == "__main__":
    logging.getLogger().setLevel(20)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--process', type=str,default='', help='check(for checking data)  pre(for processing yield data)  post(for processing raw dataset)')
    parser.add_argument('--input_root', type=str, default='./input/', help='for processing yield data')
    parser.add_argument('--output_root', type=str, default='./dataset/Without05NPY/', help='for storing processed .npy dataset')
    parser.add_argument('--data_dir', type=str, default='/mnt/d/xiaoyuwang/data/3000pixels/Combined_30m_2008_2022/', help='for storing raw .csv dataset')
    parser.add_argument('--N', type=int, default=1000, help='pixel number in each county')
    parser.add_argument('--startday', type=int, default=65, help='the start day of the growing season')
    parser.add_argument('--endday', type=int, default=278, help='134, 150, 166, 182, 198, 214, 230, 246, 262, 278')
    parser.add_argument('--startyear', type=int, default=2008, help='the start year')
    parser.add_argument('--endyear', type=int, default=2022, help='the end year')
    parser.add_argument('--usevar', type=str, default='', help='the variables used for training and testing')
    parser.add_argument('--statefp', nargs='+', default='', help='state fips')
    args = parser.parse_args()
    args.statefp = [int(i) for i in args.statefp]
    INFO(f'args:-----------------------------')
    for k,v in vars(args).items():
        INFO(f"{k}: {v}")
    INFO(f'----------------------------------')
    
    process=Process(args)
    process.run()
