
import os
import cv2
import torch
from backbone import HybridNetsBackbone

from utils.utils import Params
from pathlib import Path
import onnxruntime

import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

warnings.filterwarnings("ignore")

weight_path = 'weights/hybridnets.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device)
params = Params(
    os.path.join(Path(__file__).resolve().parent, "projects/bdd100k.yml"))
model = HybridNetsBackbone(num_classes=len(params.obj_list),
                           compound_coef=3,
                           ratios=eval(params.anchors_ratios),
                           scales=eval(params.anchors_scales),
                           seg_classes=len(params.seg_list),
                           backbone_name=None,
                           onnx_export=True)

model.load_state_dict(torch.load(weight_path, map_location=device))
model.eval()

inputs = torch.randn(1, 3, 386, 640)
print("begin to convert onnx")


torch.onnx.export(model,
                  inputs,
                  './weights/hybridnets_dynamic.onnx',
                  verbose=True,
                  opset_version=11,
                  input_names=['inputs'])
print("ONXX done")


def main(onnx_path, image_path):
    print('load from :', onnx_path)

    input = cv2.imread(image_path)
    input = cv2.resize(input, (386, 640)).transpose(2, 1, 0)
    input = input.astype(np.float32)
    input = input[np.newaxis, ...]
    # input = np.repeat(input, 3, axis=0)
    print('input', input.shape)

    # input = np.array(1, 3, 384, 640)
    teacher_session = onnxruntime.InferenceSession(onnx_path)
    soft_label = teacher_session.run([], {'inputs': input})
    print([i.shape for i in soft_label])


onnx_path = 'weights/hybridnets_dynamic.onnx'
image_path = 'images/det2.jpg'

main(onnx_path, image_path)
