exp=0



ROOT=../../../..
export PYTHONPATH=$PYTHONPATH:$ROOT

python $ROOT/validate.py \
--save_model_path=./results/epoch${e} \
--noisetype=$c \
--noisemethod=imagecorruptions \
--val_dir=/yueyuxin/data/coco/val2017 \
--dump_images=DENOISED_ONLY \
--val_ann_file=/yueyuxin/mmdetection/corruption_benchmarks/${p}/${m}/${c}-3/failure_annotation_fltest_${exp}.json \
--ckpt=results/train/model_best.pth \
2>&1|tee val.log

#--val_ann_file=/yueyuxin/mmdetection/corruption_benchmarks/retinanet/retinanet_r50_fpn_1x_coco/gaussian_noise-3/failure_annotation_fltrain_2.json \
#--ckpt=results/unet_gauss25_b4e100r02/2022-06-19-12-37/checkpoints/epoch_model_002.pth \
#--ckpt=pretrained_model/model_gauss25_b4e100r02.pth \
#--val_dir=validation/Kodak \
#--noisetype=gauss25 \
#--noisemethod=AugmentNoise \
