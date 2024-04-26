method="mlp"

mkdir -p ../output/log/${method}/
mkdir -p ../output/result/${method}/

python MIR_baseline.py \
--input_root ../dataset/100from1100_with_confidence/ \
--output_root ../output/result/${method}/ \
--model mlp \
--N 100 \
--visualize \
--testyears 2018 2019 2020 2021 2022 \
--enddays 278 \
--usevars All \
|& tee ../output/log/${method}/1.txt