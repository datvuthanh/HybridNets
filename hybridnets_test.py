import time
import torch
from torch.backends import cudnn
from backbone import HybridNetsBackbone
import cv2
import numpy as np
from glob import glob
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, boolean_string
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
parser.add_argument('--source', type=str, default='demo/image', help='The demo image folder')
parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
parser.add_argument('--nms_thresh', type=restricted_float, default='0.25')
parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
parser.add_argument('--imshow', type=boolean_string, default=False, help="Show result onscreen (unusable on colab, jupyter...)")
parser.add_argument('--imwrite', type=boolean_string, default=True, help="Write result to output folder")
parser.add_argument('--show_det', type=boolean_string, default=False, help="Output detection result exclusively")
parser.add_argument('--show_seg', type=boolean_string, default=False, help="Output segmentation result exclusively")
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
args = parser.parse_args()

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

# replace this part with your project's anchor config
anchor_ratios = [(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)]
anchor_scales = [2 ** 0, 2 ** 0.70, 2 ** 1.32]

threshold = args.nms_thresh
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

obj_list = ['car']

color_list = standard_to_bgr(STANDARD_COLORS)
ori_imgs = [cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for i in img_path]
ori_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ori_imgs]
# cv2.imwrite('ori.jpg', ori_imgs[0])
# cv2.imwrite('normalized.jpg', normalized_imgs[0]*255)
resized_shape = 640
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
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

    (input_img, _, _), ratio, pad = letterbox((input_img, input_img.copy(), input_img.copy()), resized_shape, auto=True,
                                              scaleup=False)

    input_imgs.append(input_img)
    # cv2.imwrite('input.jpg', input_img * 255)
    shapes.append(((h0, w0), ((h / h0, w / w0), pad)))  # for COCO mAP rescaling

if use_cuda:
    x = torch.stack([transform(fi).cuda() for fi in input_imgs], 0)
else:
    x = torch.stack([transform(fi) for fi in input_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16)
# print(x.shape)
model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                           ratios=anchor_ratios, scales=anchor_scales, seg_classes=2)
try:
    model.load_state_dict(torch.load(weight, map_location='cuda' if use_cuda else 'cpu'))
except:
    model.load_state_dict(torch.load(weight, map_location='cuda' if use_cuda else 'cpu')['model'])
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors, seg = model(x)

    seg = seg[:, :, 12:372, :]
    da_seg_mask = torch.nn.functional.interpolate(seg, size=[720, 1280], mode='nearest')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    for i in range(da_seg_mask.size(0)):
        #   print(i)
        da_seg_mask_ = da_seg_mask[i].squeeze().cpu().numpy().round()
        color_area = np.zeros((da_seg_mask_.shape[0], da_seg_mask_.shape[1], 3), dtype=np.uint8)
        color_area[da_seg_mask_ == 1] = [0, 255, 0]
        color_area[da_seg_mask_ == 2] = [0, 0, 255]
        color_seg = color_area[..., ::-1]
        # cv2.imwrite('seg_only_{}.jpg'.format(i), color_seg)

        color_mask = np.mean(color_seg, 2)
        # prepare to show det on 2 different imgs
        # (with and without seg) -> (full and det_only)
        det_only_imgs.append(ori_imgs[i].copy())
        seg_img = ori_imgs[i]
        seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
        seg_img = seg_img.astype(np.uint8)
        if show_seg:
          cv2.imwrite(f'{output}/{i}_seg.jpg', cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))

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

# exit()
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
