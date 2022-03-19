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
parser.add_argument('--source', type=str, default='demo/video', help='The demo video folder')
parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
parser.add_argument('--nms_thresh', type=restricted_float, default='0.25')
parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
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
video_src = glob(f'{source}/*.mp4')[0]
os.makedirs(output, exist_ok=True)
video_out = f'{output}/output.mp4'
input_imgs = []
shapes = []

# replace this part with your project's anchor config
anchor_ratios = [(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)]
anchor_scales = [2 ** 0, 2 ** 0.70, 2 ** 1.32]

threshold = args.nms_thresh
iou_threshold = args.iou_thresh

use_cuda = args.cuda
use_float16 = args.float16
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['car']

color_list = standard_to_bgr(STANDARD_COLORS)
resized_shape = 640
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
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

    (input_img, _, _), ratio, pad = letterbox((input_img, input_img.copy(), input_img.copy()), resized_shape, auto=True,
                                              scaleup=False)

    shapes = ((h0, w0), ((h / h0, w / w0), pad))

    if use_cuda:
        x = transform(input_img).cuda()
    else:
        x = transform(input_img)

    x = x.to(torch.float32 if not use_float16 else torch.float16)
    x.unsqueeze_(0)
    with torch.no_grad():
        features, regression, classification, anchors, seg = model(x)

        seg = seg[:, :, 12:372, :]
        da_seg_mask = torch.nn.functional.interpolate(seg, size=[h0, w0], mode='nearest')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask_ = da_seg_mask[0].squeeze().cpu().numpy().round()

        color_area = np.zeros((da_seg_mask_.shape[0], da_seg_mask_.shape[1], 3), dtype=np.uint8)
        color_area[da_seg_mask_ == 1] = [0, 255, 0]
        color_area[da_seg_mask_ == 2] = [0, 0, 255]
        color_seg = color_area[..., ::-1]

        # cv2.imwrite('seg_only_{}.jpg'.format(i), color_seg)

        color_mask = np.mean(color_seg, 2)
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
print("frame: {}".format(frame_count))
print("second: {}".format(t2-t1))
print("fps: {}".format((t2-t1)/frame_count))

cap.release()
out_stream.release()
