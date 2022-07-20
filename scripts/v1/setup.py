import imagecorruptions
import os
MODELS = ['retinanet_r50_fpn_1x_coco', 'retinanet_r101_fpn_1x_coco', 'faster_rcnn_r50_fpn_1x_coco', 'faster_rcnn_r101_fpn_1x_coco',
'fcos_r50_caffe_fpn_gn-head_1x_coco', 'fcos_r101_caffe_fpn_gn-head_1x_coco',
'detr_r50_8x2_150e_coco']

runsh = ''
for m in MODELS:
    f = '_r50' if '_r50' in m else '_r101'
    prefix = m[:m.find(f)]

    for c in imagecorruptions.get_corruption_names():
        for e in range(5):
            d = f'{c}-3/{m}/exp{e}'
            os.makedirs(d)
            for f in ['train.sh','quick_val.sh', 'val.sh']:
                with open(f) as fr:
                    with open(f'{d}/{f}', 'w') as fw:
                        i = 0
                        for l in fr.readlines():
                            if i==0:
                                fw.write(f'c={c}\n')
                            elif i==1:
                                fw.write(f'm={m}\n')
                            elif i==2:
                                fw.write(f'p={prefix}\n')
                            elif i==3:
                                fw.write(f'exp={e}\n')
                            else:
                                fw.write(l)
                            i+=1

