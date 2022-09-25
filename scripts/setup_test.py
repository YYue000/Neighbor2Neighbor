import os
import copy
import shutil
from mmcv import Config

test_in = 3
#test_in = 1
#test_in = 2


MODELS = ['retinanet_r50_fpn_1x_coco', 'retinanet_r101_fpn_1x_coco', 'faster_rcnn_r50_fpn_1x_coco', 'faster_rcnn_r101_fpn_1x_coco',
'fcos_r50_caffe_fpn_gn-head_1x_coco', 'fcos_r101_caffe_fpn_gn-head_1x_coco',
'detr_r50_8x2_150e_coco']
ckpt_code = {'retinanet_r50_fpn_1x_coco': '_20200130-c2398f9e',
        'faster_rcnn_r50_fpn_1x_coco': '_20200130-047c8118',
        'detr_r50_8x2_150e_coco': '_20201130_194835-2c4b8974',
        'faster_rcnn_r101_fpn_1x_coco': '_20200130-f513f705',
        'retinanet_r101_fpn_1x_coco': '_20200130-7a93545f',
        'fcos_r50_caffe_fpn_gn-head_1x_coco': '-821213aa',
        'fcos_r101_caffe_fpn_gn-head_1x_coco': '-0e37b982'}

cfg_root = '/home/yueyuxin/mmdetection/configs'
corruption_dir = '/yueyuxin/mmdetection/corruption_benchmarks/'


if __name__ == '__main__':
    corruption = 'gaussian_noise'
    supervision = 'self-supervised'
    MODELS = ['faster_rcnn_r50_fpn_1x_coco']
    #supervision = 'supervised'
    
    severity = 3

    worksp_root = f'/yueyuxin/mmdetection/repair_benchmarks/denoise-{supervision}'


    runsh = ''

    for cfg in MODELS:
        bkb = 'r101' if 'r101' in cfg else 'r50'
        prefix = cfg[:cfg.find(f'_{bkb}')]
        wkd = os.path.join(worksp_root, prefix, cfg, f'{corruption}-{severity}')

        for e in range(5):
            wkde = os.path.join(wkd, f'exp{e}') 
            #tmp
            if os.path.exists(os.path.join(wkde, 'output.bbox.json')):
                print(f'skip {wkde}')
                #continue
            os.makedirs(wkde, exist_ok=True)
            sh_list = ['test.sh', '/yueyuxin/mmdetection/repair_benchmarks/setup/get_mean.py']
            if test_in == 1:
                sh_list.append('test_clean.sh')
            for sh in sh_list:
                shutil.copy(sh, wkde)
            cfgpath = os.path.join(cfg_root, prefix, cfg+'.py')
            _cfg = Config.fromfile(cfgpath)
            _cfg['data']['test']['ann_file'] = os.path.join(corruption_dir, prefix, cfg, f'{corruption}-{severity}', f'failure_annotation_fltest_{e}.json')

            denoise_d = f'/yueyuxin/denoise/myNeighbor2Neighbor/{supervision}/{corruption}-{severity}/{cfg}/exp{e}/results/'
            test_data = _cfg['data']['test']
            _cfg['data']['test']['img_prefix'] = f'{denoise_d}/epoch/validation'
            clean_test_data = copy.deepcopy(test_data)
            clean_test_data['img_prefix'] = test_data["img_prefix"].replace("results","results_clean")
            if test_in == 2:
                _cfg['data']['test'] = dict(type='ConcatDataset', datasets=[test_data, clean_test_data])
                _cfg.dump(os.path.join(wkde, cfg+'.py'))
            elif test_in == 1:
                _cfg.dump(os.path.join(wkde, cfg+'.py'))
                _cfg['data']['test'] = clean_test_data
                _cfg.dump(os.path.join(wkde, cfg+'_cl.py'))
            elif test_in == 3:
                _cfg['data']['test'] = clean_test_data
                _cfg.dump(os.path.join(wkde, cfg+'_cl.py'))
            else:
                raise NotImplementedError

            #vals = [int(_.replace('epoch','')) for _ in os.listdir(denoise_d) if _.startswith('epoch')]
            #assert len(vals)>0, f'{denoise_d}'
            #_cfg['data']['test']['img_prefix'] = f'{denoise_d}/epoch{max(vals)}/validation'

            runsh += f'cd /yueyuxin/denoise/myNeighbor2Neighbor/{supervision}/{corruption}-{severity}/{cfg}/exp{e}\n'
            if test_in==2 or test_in==2:
                runsh += f'sh val.sh $1\nsh val_cl.sh $1\n'
            elif test_in == 3:
                runsh += f'sh val_cl.sh $1\n'
            runsh += f'cd {wkde}\n'
            ckpt = os.path.join('/yueyuxin/mmdetection/corruption_benchmarks/',prefix, cfg, f'{cfg}{ckpt_code[cfg]}.pth')
            if test_in == 2:
                runsh += f'sh test.sh {cfg}.py {ckpt} $1\npython get_mean.py --test_in 2 --MAXEPOCH 1 2>&1|tee sum.log\n'
            elif test_in == 3:
                runsh += f'sh test_clean.sh {cfg}_cl.py {ckpt} $1\npython get_mean.py --MAXEPOCH 1 2>&1|tee sum.log\n'
            elif test_in == 1:
                runsh += f'sh test.sh {cfg}.py {ckpt} $1\nsh test_clean.sh {cfg}_cl.py {ckpt} $1\npython get_mean.py --MAXEPOCH 1 2>&1|tee sum.log\n'

    with open(f'run_{corruption}_{supervision}_{len(MODELS) if len(MODELS)>1 else MODELS[0]}.sh', 'w') as fw:
        fw.write(runsh)
    
