# Learning county from pixels: Corn yield prediction with attention-weighted multiple instance learning

## 1. Data downloading

### 1.1 Download county shape file

In this paper we use 2022, you can choose the year you want.

2018 and before: [link](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.2018.html#list-tab-1556094155)

Download file: county -> cb_2018_us_county_500k 


2019 and after: [link](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html)

Download file: Counties -> 500,000 (national) shapefile [11 MB]

Unzip the .zip file and only need the .shp file

### 1.2 Download soil data file

[link](https://casoilresource.lawr.ucdavis.edu/soil-properties/download.php)

CEC: Cation Exchange Capacity

SOM: Soil Organic Matter

AWC: Avail. Water Holding Capacity

### 1.4 Run GEE code to download dataset

You need the GEE python installation first [link](https://developers.google.com/earth-engine/guides/python_install)

You need anaconda or miniconda, run:

```
conda env create -f geeenv.yaml
```
to create the conda environment.

Then run:

```
python get_data_mir.py \
--account projects/ee-xwang2696/assets/ \
--folder data \
--shp_file_path mir/cb_2016_us_county_500k \
--cec_file_path mir/cec \
--som_file_path mir/som \
--awc_file_path mir/awc \
--states 38 46 27 55 19 17 18 39 29 20 31 26 \
2>&1 | tee log.txt
```
to download the dataset. You need to modify the arguments "--account --folder --shp_file_path --cec_file_path --som_file_path --awc_file_path" based your own setting. The meaning of these parameters are shown in the argparse.

### Result: we can get the .csv file in this section

## 2. Data preprocessing

You need anaconda or miniconda, run:

```
conda env create -f attmir.yaml
```
to create the conda environment.

### 2.1 Process dataset

run:
```
python preprocess.py \
--process post \
--input_root ../input/ \
--output_root ../dataset/ \
--data_dir .././1100pixels/data/ \
--N 100 \
--startday 70 \
--endday 278 \
--startyear 2008 \
--endyear 2022 \
--usevar All \
--statefp 38 46 27 55 19 17 18 39 29 20 31 26 
```
to make a clean and usable dataset. You need to modify the arguments "--process --input_root --output_root --data_dir --N" based your own setting. The meaning of these parameters are shown in the argparse.

### Result: we can get the .npy file in this section

## 3. Train and Test

### 3.1 Run code for Attention-MIL

```
method="att"

mkdir -p ../output/log/${method}/
mkdir -p ../output/result/${method}/

python AttMIR.py \
--input_root ../dataset/100from3000/ \
--output_root ../output/result/${method}/ \
--N 100 \
--lr 0.0001 \
--epoch 500 \
--train_info 500 \
--num_heads 1 \
--hidden_size 159 \
--visualize \
--testyears 2022 \
--enddays 278 \
--usevars NDWI GCI EVI LSTday LSTnight ppt tmax tmean tmin vpdmax vpdmin awc cec som historical year \
|& tee ../output/log/${method}/1.txt
```
You need to modify the arguments "--input_root --output_root --N" based your own setting. The meaning of these parameters are shown in the argparse.

### 3.2 Run code for Instance-MIL
```
method="mlp"

mkdir -p ../output/log/${method}/
mkdir -p ../output/result/${method}/

python MIR_baseline.py \
--input_root ../dataset/2from1100/ \
--output_root ../output/result/${method}/ \
--model mlp \
--N 2 \
--visualize \
--testyears 2018 2019 2020 2021 2022 \
--enddays 278 \
--usevars All \
|& tee ../output/log/${method}/1.txt
```
You need to modify the arguments "--input_root --output_root --N" based your own setting. The meaning of these parameters are shown in the argparse.

### 3.3 Run code for Linear regression

```
method="lr"

mkdir -p ../output/log/${method}/
mkdir -p ../output/result/${method}/

python MIR_baseline.py \
--input_root ../dataset/2from1100/ \
--output_root ../output/result/${method}/ \
--model lr \
--N 2 \
--visualize \
--testyears 2018 2019 2020 2021 2022 \
--enddays 278 \
--usevars All \
|& tee ../output/log/${method}/1.txt
```
You need to modify the arguments "--input_root --output_root --N" based your own setting. The meaning of these parameters are shown in the argparse.

### 3.4 Run code for Ridge regression

```
method="ridge"

mkdir -p ../output/log/${method}/
mkdir -p ../output/result/${method}/

python MIR_baseline.py \
--input_root ../dataset/2from1100/ \
--output_root ../output/result/${method}/ \
--model ridge \
--N 2 \
--visualize \
--testyears 2018 2019 2020 2021 2022 \
--enddays 278 \
--usevars All \
|& tee ../output/log/${method}/1.txt
```
You need to modify the arguments "--input_root --output_root --N" based your own setting. The meaning of these parameters are shown in the argparse.

### 3.5 Run code for Random forest

```
method="randomforest"

mkdir -p ../output/log/${method}/
mkdir -p ../output/result/${method}/

python MIR_baseline.py \
--input_root ../dataset/2from1100/ \
--output_root ../output/result/${method}/ \
--model randomforest \
--N 2 \
--visualize \
--testyears 2018 2019 2020 2021 2022 \
--enddays 278 \
--usevars All \
|& tee ../output/log/${method}/1.txt
```
You need to modify the arguments "--input_root --output_root --N" based your own setting. The meaning of these parameters are shown in the argparse.






