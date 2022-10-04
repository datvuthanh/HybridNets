import os
import cv2
import torch
from backbone import HybridNetsBackbone
import argparse

from utils.utils import Params
from pathlib import Path
import onnxruntime
import numpy as np
import warnings
from utils.constants import *

parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                        'https://github.com/rwightman/pytorch-image-models')
parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=384)
args = parser.parse_args()

device = 'cuda' if args.cuda else 'cpu'
print('device', device)
params = Params(f'projects/{args.project}.yml')
weight = args.load_weights
weight = torch.load(weight, map_location=device)
if weight.get("optimizer"):  # strip optimizer
    weight = OrderedDict((k[6:], v) for k, v in weight['model'].items())
weight_last_layer_seg = weight['segmentation_head.0.weight']
if weight_last_layer_seg.size(0) == 1:
    seg_mode = BINARY_MODE
else:
    if params.seg_multilabel:
        seg_mode = MULTILABEL_MODE
    else:
        seg_mode = MULTICLASS_MODE
print("DETECTED SEGMENTATION MODE FROM WEIGHT AND PROJECT FILE:", seg_mode)
model = HybridNetsBackbone(num_classes=len(params.obj_list),
                           compound_coef=args.compound_coef,
                           ratios=eval(params.anchors_ratios),
                           scales=eval(params.anchors_scales),
                           seg_classes=len(params.seg_list),
                           backbone_name=args.backbone,
                           seg_mode=seg_mode,
                           onnx_export=True)

model.load_state_dict(torch.load(weight, map_location=device))
model.eval()

inputs = torch.randn(1, 3, args.height, args.width)
print("begin to convert onnx")

torch.onnx.export(model,
                  inputs,
                  'weights/hybridnets_{}x{}.onnx'.format(args.height, args.width),
                  opset_version=11,
                  input_names=['input'],
		  output_names=['regression', 'classification', 'segmentation'])
print("ONXX done")
