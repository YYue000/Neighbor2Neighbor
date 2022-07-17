ROOT=../..
export PYTHONPATH=$PYTHONPATH:$ROOT

for e in $(seq 800 10 1000)
do
python $ROOT/validate.py \
--save_model_path=./results/validation \
--noisetype=gaussian_noise \
--noisemethod=imagecorruptions \
--val_dir=/yueyuxin/data/coco/val2017 \
--dump_images=NO_DUMP \
--ckpt=results/train/2022-07-15-02-52/checkpoints/epoch_model_${e}.pth \
2>&1|tee -a val.log
done
#--val_ann_file=/yueyuxin/mmdetection/corruption_benchmarks/retinanet/retinanet_r50_fpn_1x_coco/gaussian_noise-3/failure_annotation_fltrain_2.json \
#--ckpt=results/unet_gauss25_b4e100r02/2022-06-19-12-37/checkpoints/epoch_model_002.pth \
#--ckpt=pretrained_model/model_gauss25_b4e100r02.pth \
#--val_dir=validation/Kodak \
#--noisetype=gauss25 \
#--noisemethod=AugmentNoise \
