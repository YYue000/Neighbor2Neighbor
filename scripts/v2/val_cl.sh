





ROOT=../../../..
export PYTHONPATH=$PYTHONPATH:$ROOT

python $ROOT/validate.py \
--save_model_path=./results_clean/epoch${e} \
--val_dir=/yueyuxin/data/coco/val2017 \
--val_ann_file=/yueyuxin/mmdetection/corruption_benchmarks/${p}/${m}/${c}-3/failure_annotation_fltest_${exp}.json \
--dump_images=DENOISED_ONLY \
--ckpt=results/train/model_best.pth \
--gpu=$1 \
2>&1|tee valcl.log

#--ckpt=results/unet_gauss25_b4e100r02/2022-06-19-12-37/checkpoints/epoch_model_002.pth \
#--ckpt=pretrained_model/model_gauss25_b4e100r02.pth \
#--val_dir=validation/Kodak \
#--noisetype=gauss25 \
#--noisemethod=AugmentNoise \
