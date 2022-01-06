# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

compound_coef = 1
force_input_size = None  # set None to use default size
img_path = 'datasets/bdd100k_effdet/val/b1d0a191-5490450b.jpg'

# replace this part with your project's anchor config
anchor_ratios = [(0.7, 1.4), (1.0, 1.0), (1.3, 0.8)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

# obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#             'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#             'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
#             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#             'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
#             'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
#             'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
#             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#             'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#             'toothbrush']

obj_list= ['car']


color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
ori_img = ori_imgs[0]

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load('logs/bdd100k_effdet/efficientdet-d1_0_17000.pth', map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors, seg = model(x)
    # print(ori_imgs)
    # ori_img = np.asarray(ori_imgs)
    # print(ori_img.size())
    # ratio = 640 / 1280
    # da_predict = seg[:, :, 0:(720 - 0), 0:(1280 - 0)]

    # print(da_predict.shape)
    da_seg_mask = torch.nn.functional.interpolate(seg, size = [720,1280], mode='bilinear')
    # print(da_seg_mask.shape)
    # _, da_seg_mask = torch.max(seg, 1)

    # seg = torch.rand((1, 384, 640))
    # da_seg_mask = Activation('sigmoid')(da_seg_mask)
    da_seg_mask = da_seg_mask.squeeze().cpu().numpy().round()
    # da_seg_mask[da_seg_mask < 0.5] = 0
    # da_seg_mask[da_seg_mask >= 0.5] = 1

    color_area = np.zeros((da_seg_mask.shape[0], da_seg_mask.shape[1], 3), dtype=np.uint8)

    # for label, color in enumerate(palette):
    #     color_area[result[0] == label, :] = color

    color_area[da_seg_mask == 1] = [0, 255, 0]

    color_seg = color_area[..., ::-1]

    cv2.imwrite('seg.jpg',color_seg)


    # convert to BGR
    # color_seg = color_seg[..., ::-1]
    # # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    # print(ori_img.shape)
    # ori_img = cv2.resize(ori_img, (1280, 768), interpolation=cv2.INTER_LINEAR)
    ori_img[color_mask != 0] = ori_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    ori_img = ori_img.astype(np.uint8)
    # img = cv2.resize(ori_img, (1280, 720), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('abababab.jpg', ori_img)
    # cv2.waitKey(0)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # print(x.shape)

    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)

def display(preds, imgs, imshow=True, imwrite=False):
    global color_seg
    global color_mask

    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])

        # imgs[i] = cv2.resize(imgs[i], (1280, 768), interpolation=cv2.INTER_LINEAR)
        imgs[i][color_mask != 0] = imgs[i][color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
        # imgs[i] = cv2.resize(imgs[i], (1280, 720), interpolation=cv2.INTER_LINEAR)

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])


out = invert_affine(framed_metas, out)
display(out, ori_imgs, imshow=False, imwrite=True)

print('running speed test...')
with torch.no_grad():
    print('test1: model inferring and postprocessing')
    print('inferring image for 10 times...')
    t1 = time.time()
    for _ in range(10):
        _, regression, classification, anchors, segmentation = model(x)


        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
        out = invert_affine(framed_metas, out)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    # uncomment this if you want a extreme fps test
    # print('test2: model inferring only')
    # print('inferring images for batch_size 32 for 10 times...')
    # t1 = time.time()
    # x = torch.cat([x] * 32, 0)
    # for _ in range(10):
    #     _, regression, classification, anchors = model(x)
    #
    # t2 = time.time()
    # tact_time = (t2 - t1) / 10
    # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
