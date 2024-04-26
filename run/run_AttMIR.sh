method="att"

mkdir -p ../output/log/${method}/
mkdir -p ../output/result/${method}/

# run for in-season prediction
# tune enddays, usevars is All.

python AttMIR.py \
--input_root ../dataset/100from1100_with_confidence/ \
--output_root ../output/result/${method}/ \
--N 100 \
--lr 0.001 \
--epoch 500 \
--train_info 500 \
--num_heads 1 \
--hidden_size 159 \
--visualize \
--testyears 2018 2019 2020 2021 2022 \
--enddays 278 \
--usevars All \
|& tee ../output/log/${method}/1.txt






# run for feature selection
# tune usevars, enddays is 278.

# python AttMeanMIR.py \
# --input_root ../dataset/1000from3000/ \
# --output_root ../output/result/${method}/ \
# --N 1000 \
# --lr 0.0001 \
# --epoch 500 \
# --num_heads 1 \
# --hidden_size 159 \
# --testyears 2022 \
# --enddays 278 \
# --usevars NDWI GCI EVI LSTday LSTnight ppt tmax tmean tmin vpdmax vpdmin awc cec som historical year \
# |& tee ./data/output/log/${method}/1.txt