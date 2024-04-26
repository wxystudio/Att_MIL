# statelist=[38,46,27,55,19,17,18,39,29,20,31,26]

# python get_data_mir.py \
# --account projects/ee-xwang2696/assets/ \
# --folder data \
# --states 27 55 \
# --shp_file_path /mir/cb_2016_us_county_500k \
# --cec_file_path /mir/cec \
# --som_file_path /mir/som \
# --awc_file_path /mir/awc \
# 2>&1 | tee log.txt

python get_data_mir.py \
--account projects/ee-xwang2696/assets/ \
--folder test \
--shp_file_path mir/cb_2016_us_county_500k \
--cec_file_path mir/cec \
--som_file_path mir/som \
--awc_file_path mir/awc \
--states 17 \
2>&1 | tee log.txt