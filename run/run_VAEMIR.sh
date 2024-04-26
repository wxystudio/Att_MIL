python VAEMIR.py \
--input_root ~/data/VAE_MIR_CODE/dataset/Without05NPY/ \
--output_root ~/data/VAE_MIR_CODE/output/result/VAEMIR/ \
--model_dir ~/data/VAE_MIR_CODE/ \
--N 1000 \
--batchsize 16384 \
--lr 0.001 \
--epoch 20 \
--testyears 2018 2019 2020 2021 2022 \
|& tee ~/data/VAE_MIR_CODE/output/log/VAEMIR/1.txt