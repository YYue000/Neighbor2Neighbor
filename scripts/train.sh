c=gaussian_noise
m=$1
p=$2

ROOT=../../..
export PYTHONPATH=$PYTHONPATH:$ROOT

python $ROOT/mytrain.py \
--data_root=/yueyuxin/data/coco/val2017 \
--ann_file=/yueyuxin/mmdetection/corruption_benchmarks/${p}/${m}/${c}-3/failure_annotation_fltrain_0.json \
--save_model_path=./results \
--increase_ratio=2 \
--log_freq=100 \
--log_name=train \
--dump_images=NO_DUMP \
--val_ann_file=/yueyuxin/mmdetection/corruption_benchmarks/${p}/${m}/${c}-3/failure_annotation_fltest_0.json \
--noisetype=$c \
--noisemethod=imagecorruptions \
--val_dir=/yueyuxin/data/coco/val2017 \
--n_epoch=1000 \
--n_snapshot=10 \
--n_val=1000 \
--resize_input=0 \
2>&1|tee train.log

