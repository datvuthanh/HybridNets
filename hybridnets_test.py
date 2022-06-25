import time
import torch
from torch.backends import cudnn
from backbone import HybridNetsBackbone
import cv2
import numpy as np
from glob import glob
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
from torchvision import transforms
import argparse
from utils.constants import *

parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                        'https://github.com/rwightman/pytorch-image-models')
parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
parser.add_argument('--source', type=str, default='demo/image', help='The demo image folder')
parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
parser.add_argument('--imshow', type=boolean_string, default=False, help="Show result onscreen (unusable on colab, jupyter...)")
parser.add_argument('--imwrite', type=boolean_string, default=True, help="Write result to output folder")
parser.add_argument('--show_det', type=boolean_string, default=False, help="Output detection result exclusively")
parser.add_argument('--show_seg', type=boolean_string, default=False, help="Output segmentation result exclusively")
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
parser.add_argument('--speed_test', type=boolean_string, default=False,
                    help='Measure inference latency')
args = parser.parse_args()

params = Params(f'projects/{args.project}.yml')
color_list_seg = {}
for seg_class in params.seg_list:
    # edit your color here if you wanna fix to your liking
    color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))
compound_coef = args.compound_coef
source = args.source
if source.endswith("/"):
    source = source[:-1]
output = args.output
if output.endswith("/"):
    output = output[:-1]
weight = args.load_weights
img_path = glob(f'{source}/*.jpg') + glob(f'{source}/*.png')
# img_path = [img_path[0]]  # demo with 1 image
input_imgs = []
shapes = []
det_only_imgs = []

anchors_ratios = params.anchors_ratios
anchors_scales = params.anchors_scales

threshold = args.conf_thresh
iou_threshold = args.iou_thresh
imshow = args.imshow
imwrite = args.imwrite
show_det = args.show_det
show_seg = args.show_seg
os.makedirs(output, exist_ok=True)

use_cuda = args.cuda
use_float16 = args.float16
cudnn.fastest = True
cudnn.benchmark = True

obj_list = params.obj_list
seg_list = params.seg_list

color_list = standard_to_bgr(STANDARD_COLORS)
ori_imgs = [cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for i in img_path]
ori_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ori_imgs]
print(f"FOUND {len(ori_imgs)} IMAGES")
# cv2.imwrite('ori.jpg', ori_imgs[0])
# cv2.imwrite('normalized.jpg', normalized_imgs[0]*255)
resized_shape = params.model['image_size']
if isinstance(resized_shape, list):
    resized_shape = max(resized_shape)
normalize = transforms.Normalize(
    mean=params.mean, std=params.std
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
for ori_img in ori_imgs:
    h0, w0 = ori_img.shape[:2]  # orig hw
    r = resized_shape / max(h0, w0)  # resize image to img_size
    input_img = cv2.resize(ori_img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
    h, w = input_img.shape[:2]

    (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True,
                                              scaleup=False)

    input_imgs.append(input_img)
    # cv2.imwrite('input.jpg', input_img * 255)
    shapes.append(((h0, w0), ((h / h0, w / w0), pad)))  # for COCO mAP rescaling

if use_cuda:
    x = torch.stack([transform(fi).cuda() for fi in input_imgs], 0)
else:
    x = torch.stack([transform(fi) for fi in input_imgs], 0)

x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)
# print(x.shape)
weight = torch.load(weight, map_location='cuda' if use_cuda else 'cpu')
weight_last_layer_seg = weight.get('model', weight)['segmentation_head.0.weight']
if weight_last_layer_seg.size(0) == 1:
    seg_mode = BINARY_MODE
else:
    if params.seg_multilabel:
        seg_mode = MULTILABEL_MODE
    else:
        seg_mode = MULTICLASS_MODE
print("DETECTED SEGMENTATION MODE FROM WEIGHT AND PROJECT FILE:", seg_mode)
model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=eval(anchors_ratios),
                           scales=eval(anchors_scales), seg_classes=len(seg_list), backbone_name=args.backbone,
                           seg_mode=seg_mode)
model.load_state_dict(weight.get('model', weight))

model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
    if use_float16:
        model = model.half()

with torch.no_grad():
    features, regression, classification, anchors, seg = model(x)

    # in case of MULTILABEL_MODE, each segmentation class gets their own inference image
    seg_mask_list = []
    # (B, C, W, H) -> (B, W, H)
    if seg_mode == BINARY_MODE:
        seg_mask = torch.where(seg >= 0.5, 1, 0)
        seg_mask.squeeze_(1)
        seg_mask_list.append(seg_mask)
    elif seg_mode == MULTICLASS_MODE:
        _, seg_mask = torch.max(seg, 1)
        seg_mask_list.append(seg_mask)
    else:
        seg_mask_list = [torch.where(seg[:, i, ...] >= 0.5, 1, 0) for i in range(seg.size(1))]
        # but remove background class from the list
        seg_mask_list.pop(0)
    # (B, W, H) -> (W, H)
    for i in range(seg.size(0)):
        #   print(i)
        for seg_class_index, seg_mask in enumerate(seg_mask_list):
            seg_mask_ = seg_mask[i].squeeze().cpu().numpy()
            pad_h = int(shapes[i][1][1][1])
            pad_w = int(shapes[i][1][1][0])
            seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]
            seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)
            color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
            for index, seg_class in enumerate(params.seg_list):
                    color_seg[seg_mask_ == index+1] = color_list_seg[seg_class]
            color_seg = color_seg[..., ::-1]  # RGB -> BGR
            # cv2.imwrite('seg_only_{}.jpg'.format(i), color_seg)

            color_mask = np.mean(color_seg, 2)  # (H, W, C) -> (H, W), check if any pixel is not background
            # prepare to show det on 2 different imgs
            # (with and without seg) -> (full and det_only)
            det_only_imgs.append(ori_imgs[i].copy())
            seg_img = ori_imgs[i].copy() if seg_mode == MULTILABEL_MODE else ori_imgs[i]  # do not work on original images if MULTILABEL_MODE
            seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
            seg_img = seg_img.astype(np.uint8)
            seg_filename = f'{output}/{i}_{params.seg_list[seg_class_index]}_seg.jpg' if seg_mode == MULTILABEL_MODE else \
                           f'{output}/{i}_seg.jpg'
            if show_seg or seg_mode == MULTILABEL_MODE:
                cv2.imwrite(seg_filename, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)

    for i in range(len(ori_imgs)):
        out[i]['rois'] = scale_coords(ori_imgs[i][:2], out[i]['rois'], shapes[i][0], shapes[i][1])
        for j in range(len(out[i]['rois'])):
            x1, y1, x2, y2 = out[i]['rois'][j].astype(int)
            obj = obj_list[out[i]['class_ids'][j]]
            score = float(out[i]['scores'][j])
            plot_one_box(ori_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
            if show_det:
                plot_one_box(det_only_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                             color=color_list[get_index_label(obj, obj_list)])

        if show_det:
            cv2.imwrite(f'{output}/{i}_det.jpg',  cv2.cvtColor(det_only_imgs[i], cv2.COLOR_RGB2BGR))

        if imshow:
            cv2.imshow('img', ori_imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'{output}/{i}.jpg', cv2.cvtColor(ori_imgs[i], cv2.COLOR_RGB2BGR))

if not args.speed_test:
    exit(0)
print('running speed test...')
with torch.no_grad():
    print('test1: model inferring and postprocessing')
    print('inferring 1 image for 10 times...')
    x = x[0, ...]
    x.unsqueeze_(0)
    t1 = time.time()
    for _ in range(10):
        _, regression, classification, anchors, segmentation = model(x)

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    # uncomment this if you want a extreme fps test
    print('test2: model inferring only')
    print('inferring images for batch_size 32 for 10 times...')
    t1 = time.time()
    x = torch.cat([x] * 32, 0)
    for _ in range(10):
        _, regression, classification, anchors, segmentation = model(x)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
