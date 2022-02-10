import time
from numpy.lib.type_check import imag
import torch
from torch.backends import cudnn
from matplotlib import colors
from backbone import HybridNetsBackbone
import cv2
import numpy as np
import glob
from utils.utils import letterbox, scale_coords, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, \
    plot_one_box, BBoxTransform, ClipBoxes
import os
from torchvision import transforms

compound_coef = 3
video_src = '1.mp4'
video_out = 'output.mp4'
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_stream = cv2.VideoWriter(video_out, fourcc, 30.0, (1920, 1080))
input_imgs = []
shapes = []

# replace this part with your project's anchor config
anchor_ratios = [(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)]
anchor_scales = [2 ** 0, 2 ** 0.70, 2 ** 1.32]

threshold = 0.25
iou_threshold = 0.3
imshow = False
imwrite = True

use_cuda = True
use_float16 = False
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
    model.load_state_dict(torch.load('weights/weight.pth', map_location='cuda' if use_cuda else 'cpu'), strict=False)
except:
    model.load_state_dict(torch.load('weights/weight.pth', map_location='cuda' if use_cuda else 'cpu')['model'], strict=False)
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()
cap = cv2.VideoCapture(video_src)
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
        if len(out['rois']) == 0:
            continue

        out['rois'] = scale_coords(frame[:2], out['rois'], shapes[0], shapes[1])
        for j in range(len(out['rois'])):
            x1, y1, x2, y2 = out['rois'][j].astype(np.int)
            obj = obj_list[out['class_ids'][j]]
            score = float(out['scores'][j])
            plot_one_box(frame, [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out_stream.write(frame)

cap.release()
out_stream.release()
