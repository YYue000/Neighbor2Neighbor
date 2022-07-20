c=gaussian_noise
m=$2
p=$3
exp=$4

ROOT=../../../..
export PYTHONPATH=$PYTHONPATH:$ROOT

python $ROOT/mytrain.py \
--data_root=/yueyuxin/data/coco/val2017 \
--ann_file=/yueyuxin/mmdetection/corruption_benchmarks/${p}/${m}/${c}-3/failure_annotation_fltrain_${exp}.json \
--save_model_path=./results \
--increase_ratio=2 \
--log_freq=100 \
--log_name=train \
--dump_images=NO_DUMP \
--val_ann_file=/yueyuxin/mmdetection/corruption_benchmarks/${p}/${m}/${c}-3/failure_annotation_fltest_${exp}.json \
--noisetype=$c \
--noisemethod=imagecorruptions \
--val_dir=/yueyuxin/data/coco/val2017 \
--n_epoch=1000 \
--n_snapshot=10000 \
--n_val=10 \
--st_val=600 \
--resize_input=0 \
--gpu_devices=$1 \
2>&1|tee train.log

