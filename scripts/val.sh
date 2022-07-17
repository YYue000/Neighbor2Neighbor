ROOT=../..
export PYTHONPATH=$PYTHONPATH:$ROOT

e=800
python $ROOT/validate.py \
--save_model_path=./results/epoch${e} \
--noisetype=gaussian_noise \
--noisemethod=imagecorruptions \
--val_dir=/yueyuxin/data/coco/val2017 \
--dump_images=DENOISED_ONLY \
--ckpt=results/train/2022-07-03-13-01/checkpoints/epoch_model_${e}.pth \
2>&1|tee val_${e}.log

#--val_ann_file=/yueyuxin/mmdetection/corruption_benchmarks/retinanet/retinanet_r50_fpn_1x_coco/gaussian_noise-3/failure_annotation_fltrain_2.json \
#--ckpt=results/unet_gauss25_b4e100r02/2022-06-19-12-37/checkpoints/epoch_model_002.pth \
#--ckpt=pretrained_model/model_gauss25_b4e100r02.pth \
#--val_dir=validation/Kodak \
#--noisetype=gauss25 \
#--noisemethod=AugmentNoise \
