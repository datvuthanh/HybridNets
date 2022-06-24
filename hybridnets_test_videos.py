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
parser.add_argument('--source', type=str, default='demo/video', help='The demo video folder')
parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
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
video_srcs = glob(f'{source}/*.mp4')
os.makedirs(output, exist_ok=True)
input_imgs = []
shapes = []

anchors_ratios = params.anchors_ratios
anchors_scales = params.anchors_scales

threshold = args.conf_thresh
iou_threshold = args.iou_thresh

use_cuda = args.cuda
use_float16 = args.float16
cudnn.fastest = True
cudnn.benchmark = True

obj_list = params.obj_list
seg_list = params.seg_list

color_list = standard_to_bgr(STANDARD_COLORS)
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
# print(x.shape)
weight = torch.load(weight, map_location='cuda' if use_cuda else 'cpu')
weight_last_layer_seg = weight.get('model', weight)['segmentation_head.0.weight']
if weight_last_layer_seg.size(0) == 1:
    seg_mode = BINARY_MODE
else:
    if params.seg_multilabel:
        seg_mode = MULTILABEL_MODE
        print("Sorry, we do not support multilabel video inference yet.")
        print("In image inference, we can give each class their own image.")
        print("But a video for each class is meaningless.")
        print("https://github.com/datvuthanh/HybridNets/issues/20")
        exit(0)
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

for video_index, video_src in enumerate(video_srcs):
    video_out = f'{output}/{video_index}.mp4'
    cap = cv2.VideoCapture(video_src)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_stream = cv2.VideoWriter(video_out, fourcc, 30.0,
                                 (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    t1 = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h0, w0 = frame.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        input_img = cv2.resize(frame, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
        h, w = input_img.shape[:2]

        (input_img, _), ratio, pad = letterbox((input_img, None), auto=False,
                                                  scaleup=True)

        shapes = ((h0, w0), ((h / h0, w / w0), pad))

        if use_cuda:
            x = transform(input_img).cuda()
        else:
            x = transform(input_img)

        x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)
        x.unsqueeze_(0)
        with torch.no_grad():
            features, regression, classification, anchors, seg = model(x)

            seg = seg[:, :, int(pad[1]):int(h+pad[1]), int(pad[0]):int(w+pad[0])]
            # (1, C, W, H) -> (1, W, H)
            if seg_mode == BINARY_MODE:
                seg_mask = torch.where(seg >= 0.5, 1, 0)
                seg_mask.squeeze_(1)
            else:
                _, seg_mask = torch.max(seg, 1)
            # (1, W, H) -> (W, H)
            seg_mask_ = seg_mask[0].squeeze().cpu().numpy()
            seg_mask_ = cv2.resize(seg_mask_, dsize=(w0, h0), interpolation=cv2.INTER_NEAREST)
            color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
            for index, seg_class in enumerate(params.seg_list):
                color_seg[seg_mask_ == index+1] = color_list_seg[seg_class]
            color_seg = color_seg[..., ::-1]  # RGB -> BGR
            # cv2.imwrite('seg_only_{}.jpg'.format(i), color_seg)

            color_mask = np.mean(color_seg, 2)  # (H, W, C) -> (H, W), check if any pixel is not background
            frame[color_mask != 0] = frame[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
            frame = frame.astype(np.uint8)
            # cv2.imwrite('seg_{}.jpg'.format(i), ori_img)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)
            out = out[0]
            out['rois'] = scale_coords(frame[:2], out['rois'], shapes[0], shapes[1])
            for j in range(len(out['rois'])):
                x1, y1, x2, y2 = out['rois'][j].astype(int)
                obj = obj_list[out['class_ids'][j]]
                score = float(out['scores'][j])
                plot_one_box(frame, [x1, y1, x2, y2], label=obj, score=score,
                             color=color_list[get_index_label(obj, obj_list)])
            out_stream.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1

    t2 = time.time()
    print("video: {}".format(video_src))
    print("frame: {}".format(frame_count))
    print("second: {}".format(t2-t1))
    print("fps: {}".format((t2-t1)/frame_count))

    cap.release()
    out_stream.release()
