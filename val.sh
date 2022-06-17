python validate.py \
--save_model_path=./results \
--val_dir=debug \
--ckpt=pretrained_model/model_gauss25_b4e100r02.pth \
--noisetype=gauss25 \
--noisemethod=AugmentNoise \
2>&1|tee val.log
#--dump_denoise_only \
#--noisetype=gaussian_noise \
#--noisemethod=imagecorruptions \
#--val_dir=/yueyuxin/data/coco/val2017 \

