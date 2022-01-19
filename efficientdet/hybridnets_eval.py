# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import argparse

import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string, ConfusionMatrix, scale_coords, process_batch, \
                ap_per_class

from backbone import EfficientDetBackbone
from efficientdet.bdd import BddDataset
from efficientdet.AutoDriveDataset import AutoDriveDataset
from efficientdet.yolop_cfg import update_config
from efficientdet.yolop_cfg import _C as cfg
from efficientdet.yolop_utils import DataLoaderX
from torchvision import transforms
import numpy as np
from pathlib import Path
import utils.smp_metrics as smp_metrics

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5,
                help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights

print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']
print(params)
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef],
                                                         mean=params['mean'], std=params['std'])
        print(framed_imgs[0].shape)
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)

        features, regression, classification, anchors, segmentation = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()





if __name__ == '__main__':
    SET_NAME = params['val_set']
    VAL_GT = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params["project_name"]}/{SET_NAME}/'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    obj_list = ['car']

    valid_dataset = BddDataset(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])
    )

    val_generator = DataLoaderX(
        valid_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=AutoDriveDataset.collate_fn
    )

    if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(params['obj_list']),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']),
                                     seg_classes = len(params['seg_list']))
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        except:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['model'])
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model.cuda(gpu)

            if use_float16:
                model.half()

        loss_regression_ls = []
        loss_classification_ls = []
        loss_segmentation_ls = []
        jdict, stats, ap, ap_class = [], [], [], []
        iou_thresholds = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
        num_thresholds = iou_thresholds.numel()
        nc = 1
        seen = 0
        plots = True
        confusion_matrix = ConfusionMatrix(nc=nc)
        s = ('%15s' + '%11s' * 12) % (
            'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'mIoU', 'mF1', 'rIoU', 'rF1', 'lIoU', 'lF1')
        dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        iou_ls = [[] for _ in range(3)]
        f1_ls = [[] for _ in range(3)]
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        val_loader = tqdm(val_generator)
        for iter, data in enumerate(val_loader):
            with torch.no_grad():
                imgs = data['img']
                annot = data['annot']
                seg_annot = data['segmentation']

                # if params.num_gpus == 1:
                imgs = imgs.cuda()
                annot = annot.cuda()
                seg_annot = seg_annot.cuda()

            features, regressions, classifications, anchors, segmentation = model(imgs)

            out = postprocess(imgs.detach(),
                                          torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(),
                                          classifications.detach(),
                                          regressBoxes, clipBoxes,
                                          0.5, 0.3)

            framed_metas = [[640, 384, 1280, 720, 0, 0] for _ in range(len(out))]

            out = invert_affine(framed_metas, out)

            for i in range(annot.size(0)):
                seen += 1
                labels = annot[i]
                labels = labels[labels[:, 4] != -1]

                ou = out[i]
                nl = len(labels)

                pred = np.column_stack([ou['rois'], ou['scores']])
                pred = np.column_stack([pred, ou['class_ids']])
                pred = torch.from_numpy(pred).cuda()

                target_class = labels[:, 4].tolist() if nl else []  # target class

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, num_thresholds, dtype=torch.bool),
                                      torch.Tensor(), torch.Tensor(), target_class))
                    # print("here")
                    continue

                if nl:
                    labels = scale_coords((384, 640), labels, (720, 1280))
                    correct = process_batch(pred, labels, iou_thresholds)
                    if plots:
                        confusion_matrix.process_batch(pred, labels)
                else:
                    correct = torch.zeros(pred.shape[0], num_thresholds, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))


                # print(stats)

                # Visualization
                # seg_0 = segmentation[i]
                # # print('bbb', seg_0.shape)
                # seg_0 = torch.argmax(seg_0, dim = 0)
                # # print('before', seg_0.shape)
                # seg_0 = seg_0.cpu().numpy()
                #     #.transpose(1, 2, 0)
                # # print(seg_0.shape)
                # anh = np.zeros((384,640,3))
                # anh[seg_0 == 0] = (255,0,0)
                # anh[seg_0 == 1] = (0,255,0)
                # anh[seg_0 == 2] = (0,0,255)
                # anh = np.uint8(anh)
                # cv2.imwrite('segmentation-{}.jpg'.format(filenames[i]),anh)

            for i in range(len(params['seg_list'])+1):
                # print(segmentation[:,i,...].unsqueeze(1).size())
                tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(segmentation[:,i,...].unsqueeze(1).cuda(),
                                                                        seg_annot[:, i, ...].unsqueeze(1).round().long().cuda(),
                                                                        mode='binary', threshold=0.5)

                iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg).mean()
                # print("I", i , iou)
                f1 = smp_metrics.f1_score(tp_seg, fp_seg, fn_seg, tn_seg).mean()

                iou_ls[i].append(iou.detach().cpu().numpy())
                f1_ls[i].append(f1.detach().cpu().numpy())

        # print(len(iou_ls[0]))
        # print(iou_ls)
        iou_score = np.mean(iou_ls)
        # print(iou_score)
        f1_score = np.mean(f1_ls)

        for i in range(len(params['seg_list']) + 1):
            iou_ls[i] = np.mean(iou_ls[i])
            f1_ls[i] = np.mean(f1_ls[i])

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        # print(stats[3])

        # Count detected boxes per class
        # boxes_per_class = np.bincount(stats[2].astype(np.int64), minlength=1)

        ap50 = None
        save_dir = 'abc'
        names = {
            0: 'car'
        }
        # Compute metrics
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=1)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        print(s)
        pf = '%15s' + '%11i' * 2 + '%11.3g' * 10  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map, iou_score, f1_score,
                    iou_ls[1], f1_ls[1], iou_ls[2], f1_ls[2]))

        # Print results per class
        verbose = True
        training = False
        nc = 1
        if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Plots
        if plots:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
            confusion_matrix.tp_fp()