import pdb
import json
import os
import sys
import pdb
import argparse
import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import error as ERROR
logging.basicConfig(level=20, format="[%(levelname)s: %(filename)s line:%(lineno)d] %(message)s")
import warnings
warnings.filterwarnings("ignore")
import ee
import time

from tqdm import tqdm
# import geemap

# from comosites_geemap import *
# from comosites_geemap import stackCollection, appendBand


def stackCollection(collection):
  return ee.Image(collection.iterate(appendBand))

def appendBand(current, previous):
  # Append it to the result (only return current item on first element)
  accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, ee.Image(previous).addBands(ee.Image(current)))
  # return the accumulation
  return accum

def updateMask(img):
  return img.updateMask(cropMask)

#*
 # Export county-averaged EVI, GCI, NDWI value from
 # NBAR (MCD43A4) collection
 #
def getEVI(image):
  evi = image.expression(
      '2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 10000)',
        {
          'red': image.select([0]).float(),    # 620-670nm, RED
          'nir': image.select([1]).float(),    # 841-876nm, NIR
          'blue': image.select([2]).float()   # 459-479nm, BLUE
        })

  return evi.updateMask(evi.gt(0)).updateMask(evi.lt(1))

def addEVI(image):
  evi = image.expression(
      '2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 10000)',
        {
          'red': image.select([0]).float(),    # 620-670nm, RED
          'nir': image.select([1]).float(),    # 841-876nm, NIR
          'blue': image.select([2]).float()   # 459-479nm, BLUE
        })

  return image.addBands(evi.updateMask(evi.gt(0)).updateMask(evi.lt(1)).unmask().rename('EVI'))
  #*
   # Calculate GCI
   #
def getGCI(image):
  gci = image.expression(
    'nir / green - 1', {
      'nir': image.select([1]).float(),
      'green': image.select([3]).float()
    })
  return gci.updateMask(gci.gt(0))

def addGCI(image):
  gci = image.expression(
    'nir / green - 1', {
      'nir': image.select([1]).float(),
      'green': image.select([3]).float()
    })
  return image.addBands((gci.updateMask(gci.gt(0))).unmask().rename('GCI'))


  #*
   # Calculate NDWI
   #
def getNDWI(image):
  ndwi = image.expression(
    '(nir - swir) / (nir + swir)', {
      'nir': image.select([1]).float(),
      'swir': image.select([4]).float()
    })
  return ndwi.updateMask(ndwi.gt(-1)).updateMask(ndwi.lt(1))

def addNDWI(image):
  ndwi = image.expression(
    '(nir - swir) / (nir + swir)', {
      'nir': image.select([1]).float(),
      'swir': image.select([4]).float()
    })
  return image.addBands(ndwi.updateMask(ndwi.gt(-1)).updateMask(ndwi.lt(1)).unmask().rename('NDWI'))

def getNDVI(image):
  ndvi = image.expression(
    '(nir - red) / (nir + red)', {
      'nir': image.select([1]).float(),
      'red': image.select([0]).float()
    })
  return ndvi.updateMask(ndvi.gt(-1)).updateMask(ndvi.lt(1))

def addNDVI(image):
  ndvi = image.expression(
    '(nir - red) / (nir + red)', {
      'nir': image.select([1]).float(),
      'red': image.select([0]).float()
    })
  return image.addBands(ndvi.updateMask(ndvi.gt(-1)).updateMask(ndvi.lt(1)).unmask().rename('NDVI'))


  #*
   # Function to add "DOY" as bandname
   #
def addBandname(img):
    datestring = ee.String(img.get('system:index'))
    format = 'YYYY_MM_dd'
    eedate = ee.Date.parse(format, datestring)
    doy = ee.Number(eedate.getRelative('day', 'year').add(1)).format()
    year = ee.Number(eedate.get('year')).format()
    bandname = year.cat('_').cat(doy)
    return img.set('bandname',bandname)
    
    # Create a list of "daily" features to join the GLDAS images to
  # Assign a Date string property for joining/grouping by day
def daily_func(date):
  return ee.Feature(
    None,
    {
      'DATE': ee.Date(date).format('YYYY-MM-dd'),
      'DOY': ee.Date(date).getRelative('day','year').add(1),
      'system:time_start': ee.Number(date),
      'system:time_end': ee.Number(date).add(24*60*60*1000)
    })

def set_date_func(obj):
  date = ee.Date(obj.get('system:time_start'))
  return obj.set({
    'DATE': date.format('YYYY-MM-dd'),
    'DOY': date.getRelative('day','year').add(1)
  })
# Compute daily GLDAS
def gldas_daily_func(ft):

  # Get joined images for each day of year
  gcoll = ee.ImageCollection.fromImages(ft.get('gldas_images'))

  # Get average
  return gcoll.mean().set({
    'DATE': ft.get('DATE'),
    'DOY': ft.get('DOY')
  })
  
def GLDASaddDOY(img):
  doy = img.get('DOY')
  
  def func_plp(name):
    return ee.String(name).cat('_').cat(ee.Number(doy).format('%d'))
    # get bandnames
  names = img.bandNames().map(func_plp)
  return img.select(ee.List.sequence(0,None,1,names.length()),names)
  
def addDOY(img):

  datestring = ee.String(img.get('system:index'))
  format = 'YYYY_MM_dd'
  eedate = ee.Date.parse(format, datestring)
  doy =eedate.getRelative('day', 'year').add(1)
  # get bandnames
  names = img.bandNames().map(lambda name : ee.String(name).cat('_').cat(ee.Number(doy).format('%d')))
  return img.select(ee.List.sequence(0,None,1,names.length()),names) \
            .set('DOY',doy)
def PRISMaddDOY(img):

  datestring = ee.String(img.get('system:index'))
  format = 'YYYYMMdd'
  eedate = ee.Date.parse(format, datestring)
  doy =eedate.getRelative('day', 'year').add(1)
  # get bandnames
  names = img.bandNames().map(lambda name : ee.String(name).cat('_').cat(ee.Number(doy).format('%d')))
  return img.select(ee.List.sequence(0,None,1,names.length()),names) \
            .set('DOY',doy)

def stackLST(start,end,region,mask,mode):
  # mode: 'Day' or 'Night'
  band = 'LST_' + mode + '_1km'
  qcband = 'QC_' + mode
  def addBandname2(img):
    datestring = ee.String(img.get('system:index'))
    format = 'YYYY_MM_dd'
    eedate = ee.Date.parse(format, datestring)
    doy = ee.Number(eedate.getRelative('day', 'year').add(1)).format()
    year = ee.Number(eedate.get('year')).format()
    bandname = ee.String(band).cat('_').cat(doy)
    return img.set('bandname',bandname)
  LST = ee.ImageCollection('MODIS/006/MYD11A1') \
              .filterDate(start,end) \
              .filterBounds(region)\
              .map(lambda img:img.updateMask(img.select(qcband).eq(0)))\
                .map(lambda img:img.select(band).float().multiply(0.02).updateMask(mask) )\
              .map(addBandname2) \
              .sort('DOY',True)
  # print(start,end,region,LST)
  LSTstack = stackCollection(LST)
  return LSTstack

def stackPRISM(start, end, region, mask, feature_type):
  prism = ee.ImageCollection('OREGONSTATE/PRISM/AN81d') \
              .filterDate(start, end) \
              .filterBounds(region) \
              .select(feature_type) \
              .map(PRISMaddDOY)\
                .map(lambda img: img.updateMask(mask))\
              .iterate(appendBand)
  return ee.Image(prism)

def exportVI(table, prefix, folder):
  task=ee.batch.Export.table.toDrive(
  collection=table,
  description= prefix,
  folder= folder,  #*********
  fileNamePrefix=prefix)
  return task

def get_account(num):
    if num == 0:
        account = 'users/xiaoyuwangxjtu/'
    if num == 1:
        account = 'projects/ee-wxystudiogee1/assets/'
    if num == 2:
        # account = 'projects/ee-wxystudiogee2/assets/'
        account = 'projects/ee-xwang2696/assets/'
    if num == 3:
        account = 'projects/ee-wxystudiogee3/assets/'
    if num == 4:
        account = 'projects/ee-wxystudiogee4/assets/'
    return account

def count_num(statelist, counties):
   for fp in statelist:
        count = 0
        temp=counties.filter(ee.Filter.eq('STATEFP',str(fp).rjust(2,'0')))
        INFO(f"state: {fp}")
        if(temp.size().getInfo()==0):
            continue
        else:
            countylist=temp.aggregate_array('COUNTYFP').getInfo()
            # start=countylist.index('219')
            INFO(f"countylist: {countylist}")
            # pdb.set_trace()
            for i,county in enumerate(countylist):
                years=ee.List.sequence(2008,2022).getInfo()
                INFO(f"years: {years}")
                # pdb.set_trace()
                # years=ee.List([2001,2008]).getInfo()
                for year in years:
                   count += 1
        INFO(f"count: {count}")


def getmiss():
    miss_list = []
    with open('missing_data.txt') as f:
        for line in f:
            INFO(f"line: {line}")
            miss_list.append(line.strip())
    return miss_list

if __name__ == '__main__':
    logging.getLogger().setLevel(30)
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", type=str, help='account name of gee gloud assets')
    parser.add_argument("--folder", type=str, help='folder name: omit(for some omitted data), test(for test code), data((for downloading data))')
    parser.add_argument("--shp_file_path", type=str, help='county shape file path in gee gloud assets')
    parser.add_argument("--cec_file_path", type=str, help='cec file path in gee gloud assets')
    parser.add_argument("--som_file_path", type=str, help='som file path in gee gloud assets')
    parser.add_argument("--awc_file_path", type=str, help='awc file path in gee gloud assets')
    parser.add_argument("--states",nargs='+',default='',help="states fips")
    args = parser.parse_args()
    args.states = [int(i) for i in args.states]
    INFO(f'args:-----------------------')
    for k,v in vars(args).items():
        INFO(f'{k}: {v}')
    INFO(f'----------------------------')
    # pdb.set_trace()

    ee.Authenticate()
    ee.Initialize()

    cec=ee.Image(args.account+args.cec_file_path)
    som=ee.Image(args.account+args.som_file_path)
    awc=ee.Image(args.account+args.awc_file_path)
    UScounties = ee.FeatureCollection(args.account+args.shp_file_path)

    INFO(f"UScounties: {type(UScounties)} {type(UScounties.getInfo())}")
    INFO(f"size: {type(UScounties.size())} {type(UScounties.size().getInfo())}")
    INFO(f"state: {UScounties.limit(10).aggregate_array('STATEFP').getInfo()}")
    INFO(f"county: {UScounties.limit(10).aggregate_array('COUNTYFP').getInfo()}")
    INFO(f"propertyNames: {type(UScounties.first().propertyNames())} {type(UScounties.first().propertyNames().getInfo())} {UScounties.first().propertyNames().getInfo()}")
    INFO(f"propertyNames: {type(UScounties.propertyNames())} {type(UScounties.propertyNames().getInfo())} {UScounties.propertyNames().getInfo()}")
    
    download_list = []
    with open('download_list.txt', 'r') as f:
        for line in f:
            download_list.append(line)
    INFO(f"download_list: {download_list}")

    # count_num(states, UScounties)
    if(args.folder == 'omit'):               
      miss_list = getmiss()
    # INFO(f"miss_list: {miss_list}")
    # pdb.set_trace()



    for fp in args.states:
        filtered_state = UScounties.filter(ee.Filter.eq('STATEFP',str(fp).rjust(2,'0')))
        INFO(f"fp: {fp}")
        INFO(f"filtered_state: {type(filtered_state)} {filtered_state.size().getInfo()}")
        if(filtered_state.size().getInfo()==0):
            raise ValueError(f"state {fp} is empty")
        countylist=filtered_state.aggregate_array('COUNTYFP').getInfo()
        INFO(f"countylist: {countylist}")
        # pdb.set_trace()
        for i,county in enumerate(countylist):
            filtered_county=filtered_state.filter(ee.Filter.eq('COUNTYFP',str(county).rjust(3,'0')))
            years=ee.List.sequence(2008,2022).getInfo()
            INFO(f"years: {years}")
            # pdb.set_trace()
            for year in tqdm(years):
                INFO(f"fp: {type(fp)} {fp} county: {type(county)} {county} year: {type(year)} {year}")
                if(args.folder == 'omit'):
                  if(str(year)+'_'+str(fp)+'_'+str(county) in miss_list):
                      pass
                  else:
                      continue
                elif(args.folder == 'data'):
                  download_index = str(fp)+'_'+str(county)+'_'+str(year)
                  INFO(f"download_index: {download_index}")
                  if(download_index+'\n' in download_list):
                      if(download_index != download_list[-1]):
                          continue
                  else:
                      download_list.append(f"{download_index}")
                      with open('download_list.txt', 'a+') as f:
                          f.write(download_index+'\n')
                
                start_day = str(year) + '-03-01'
                end_day = str(year) + '-10-31'
                if year>2006:
                    dataset = ee.ImageCollection('USDA/NASS/CDL') \
                                    .filter(ee.Filter.date( str(year) + '-01-01', str(year) + '-12-31')) \
                                    .first()
                    cropMask = dataset.select('cropland').eq(1); #1 - corn, 5 -soybeans
                    # confidence = dataset.select('confidence')
                else:
                    mcdband = 'MODIS/006/MCD12Q1/' + str(year) + '_01_01'
                    # mcdband = ee.ImageCollection('MODIS/006/MCD12Q1').filter(ee.Filter.date(start_day, end_day))
                    cropMask = ee.Image(mcdband).select('LC_Type1').clip(filtered_county).eq(12)
                
                INFO(f'dataset: {type(dataset)}')
                INFO(f"cropMask: {type(cropMask)}")
                INFO(f'bandNames: {cropMask.bandNames().getInfo()}')

                modVI = ee.ImageCollection('MODIS/006/MCD43A4') \
                        .filterDate(start_day, end_day) \
                        .filterBounds(filtered_county) \
                        .map(updateMask)\
                        .map(addNDVI)\
                        .map(addNDWI).map(addEVI).map(addGCI)\
                        .select(['NDWI','EVI','GCI'])\
                        .map(addDOY)
                INFO(f"modVI: {type(modVI)}")
                modVIStack = ee.Image(modVI.iterate(appendBand)).clip(filtered_county)
                INFO(f"modVIStack: {type(modVIStack)}")
                INFO(f"bandNames: {modVIStack.bandNames().getInfo()}")

                proj_modis = modVIStack.select('EVI_61').projection()
                proj_cropmask = cropMask.projection()
                # modVIStack = modVIStack.reproject('EPSG:5070', None, proj_modis.nominalScale())
                # proj_modis = modVIStack.select('EVI_61').projection()

                Comgldas=modVIStack
                #############SSURGO###################
                Comsoil=Comgldas.addBands(cec.unmask().rename('cec').clip(filtered_county))###add soil band
                Comsoil=Comsoil.addBands(som.unmask().rename('som').clip(filtered_county))###add soil band
                Comsoil=Comsoil.addBands(awc.unmask().rename('awc').clip(filtered_county))###add soil band
                INFO(f"Comsoil: {type(Comsoil)}")
                #############LST###################
                LSTday_stack = stackLST(start_day, end_day, filtered_county, cropMask, 'Day')
                INFO(f"LSTday_stack: {type(LSTday_stack)}")
                LSTnight_stack = stackLST(start_day, end_day, filtered_county, cropMask, 'Night')
                INFO(f"LSTnight_stack: {type(LSTnight_stack)}")
                ComLST=Comsoil.addBands(LSTday_stack.unmask())
                ComLST=ComLST.addBands(LSTnight_stack.unmask())
                INFO(f"ComLST: {type(ComLST)}")
                #############PRISM###################
                # # get prism collection
                # ppt = ['ppt']
                # prism_ppt = stackPRISM(start_day,end_day, filtered_county, cropMask, ppt)

                # # get prism temp
                # temp = ['tmin','tmean','tmax']
                # prism_temp = stackPRISM(start_day,end_day, filtered_county, cropMask, temp)

                # # get prism vpd
                vpd = ['vpdmin','vpdmax','tmin','tmean','tmax','ppt']
                prism_vpd = stackPRISM(start_day,end_day, filtered_county, cropMask, vpd)
                INFO(f"prism_vpd: {type(prism_vpd)}")
                ComPrism=ComLST.addBands(prism_vpd.unmask())
                INFO(f"ComPrism: {type(ComPrism)}")
                ###############assign tasks#################################
                final=ComPrism.updateMask(cropMask)
                INFO(f"final: {type(final)}")
                tileScale=4
                scale=30
                numPixels=1100
                f=final.sample(filtered_county, scale, numPixels=numPixels,tileScale=tileScale,geometries=True)
                # INFO(f'{i} {f.size().getInfo()}')
                ERROR(f"fp: {fp} county: {i}/{len(countylist)} years: {year} exportfile: {'Data_' + str(year)+'_STATE_'+str(fp).rjust(2,'0')+'_COUNTY_'+str(county).rjust(3,'0')}")

                rectangle = ee.Geometry.Rectangle([-70, 25, -130, 49])
                grid_pol_modis = rectangle.coveringGrid(proj_modis)
                grid_pol_cropmask = rectangle.coveringGrid(proj_cropmask)
                def map_function1(ele):
                   return grid_pol_modis.filterBounds(ee.Feature(ele).geometry()).first()
                
                modisPixels = f.map(map_function1)
                # INFO(f"modisPixels: {modisPixels.getInfo().keys()}")
                image = cropMask.clip(modisPixels)
                countByCropMask = image.reduceRegions(modisPixels, ee.Reducer.frequencyHistogram(), 30)
                # INFO(f"countByCropMask: {countByCropMask.getInfo()} {countByCropMask.size().getInfo()}")
                list = countByCropMask.toList(countByCropMask.size())
                # INFO(f"list: {list.getInfo()}")
                def map_function2(ele):
                   keys = ee.Dictionary(ee.Feature(ele).get('histogram')).keys()
                   size = keys.size()
                   values = ee.Dictionary(ee.Feature(ele).get('histogram')).values()
                   opt = ee.Algorithms.If(size.eq(1),[values.get(0),0],[values.get(0),values.get(1)])
                   return ee.Number(ee.List(opt).get(1)).format('%.0f')
                
                count = list.map(map_function2)
                # INFO(f"count: {count.getInfo()}")
                f_list = f.toList(f.size())
                def map_function3(feat):
                   idx = ee.List(f_list).indexOf(feat)
                   return feat.set('confidence', ee.List(count).get(idx))
                   
                f = f.map(map_function3)

                # flag=True
                # while flag:
                #    if  task.status()['state']!='COMPLETED' and task.status()['state']!='FAILED':
                #       time.sleep(2)
                #    elif task.status()['state']=='FAILED':
                #       ERROR(f"FAILED: {numPixels}")
                #       numPixels-=1000
                #       f=final.sample(filtered_county, scale, numPixels=numPixels,tileScale=tileScale)
                #       # f=final.sampleRegions(filtered_county,scale=scale,geometries=False,tileScale=tileScale)
                #       task=exportVI(f, 'Data_' + str(year)+'_STATE_'+str(fp).rjust(2,'0')+'_COUNTY_'+str(county).rjust(3,'0'))
                #       task.start()
                #    else:
                #       # print(tileScale)
                #       flag=False

                if(args.folder == 'data'):
                  task=exportVI(f, 'Data_' + str(year)+'_STATE_'+str(fp).rjust(2,'0')+'_COUNTY_'+str(county).rjust(3,'0'), 'data')
                elif(args.folder == 'omit'):
                  task=exportVI(f, 'Data_' + str(year)+'_STATE_'+str(fp).rjust(2,'0')+'_COUNTY_'+str(county).rjust(3,'0'), 'omit')
                elif(args.folder == 'test'):
                  task=exportVI(f, 'Data_' + str(year)+'_STATE_'+str(fp).rjust(2,'0')+'_COUNTY_'+str(county).rjust(3,'0'), 'test')
                else:
                    raise Exception('Invalid folder name')
                task.start()

                if(args.folder == 'test'):
                  pdb.set_trace()

            
            