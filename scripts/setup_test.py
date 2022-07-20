import os
import shutil
from mmcv import Config

MODELS = ['retinanet_r50_fpn_1x_coco', 'retinanet_r101_fpn_1x_coco', 'faster_rcnn_r50_fpn_1x_coco', 'faster_rcnn_r101_fpn_1x_coco',
'fcos_r50_caffe_fpn_gn-head_1x_coco', 'fcos_r101_caffe_fpn_gn-head_1x_coco',
'detr_r50_8x2_150e_coco']


cfg_root = '/home/yueyuxin/mmdetection/configs'
corruption_dir = '/yueyuxin/mmdetection/corruption_benchmarks/'


if __name__ == '__main__':
    corruption = 'gaussian_noise'
    supervision = 'self-supervised'
    #supervision = 'supervised'
    
    severity = 3
    sh_path = f'test.sh'

    worksp_root = f'/yueyuxin/mmdetection/repair_benchmarks/denoise-{supervision}'


    runsh = ''

    for cfg in MODELS:
        bkb = 'r101' if 'r101' in cfg else 'r50'
        prefix = cfg[:cfg.find(f'_{bkb}')]
        wkd = os.path.join(worksp_root, prefix, cfg, f'{corruption}-{severity}')

        for e in range(5):
            wkde = os.path.join(wkd, f'exp{e}') 
            if not os.path.exists(wkde):
                os.makedirs(wkde)
                shutil.copy(sh_path, wkde)
            cfgpath = os.path.join(cfg_root, prefix, cfg+'.py')
            _cfg = Config.fromfile(cfgpath)
            _cfg['data']['test']['ann_file'] = os.path.join(corruption_dir, prefix, cfg, f'{corruption}-{severity}', f'failure_annotation_fltest_{e}.json')

            denoise_d = f'/yueyuxin/denoise/myNeighbor2Neighbor/{supervision}/{corruption}-{severity}/{cfg}/exp{e}/results/'
            vals = [int(_.replace('epoch','')) for _ in os.listdir(denoise_d) if _.startswith('epoch')]
            assert len(vals)>0, f'{denoise_d}'
            _cfg['data']['test']['img_prefix'] = f'{denoise_d}/epoch{max(vals)}/validation'
            _cfg.dump(os.path.join(wkde, cfg+'.py'))


            runsh += f'cd {wkde}\n'
            runsh += f'sh test.sh {cfg}.py  $1\n'
            xx

    with open(f'run_{corruption}_{supervision}.sh', 'w') as fw:
        fw.write(runsh)
    
