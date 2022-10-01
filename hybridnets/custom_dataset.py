import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset
from utils.utils import letterbox, augment_hsv, random_perspective, box_candidates, mixup
from tqdm.autonotebook import tqdm
import json
import albumentations as A
from collections import OrderedDict
from utils.constants import *
import torchshow
from hybridnets.dataset import BddDataset
import json
from skimage.draw import polygon


class CustomDataset(BddDataset):
    def __init__(self, params, is_train, inputsize=[640, 384], transform=None, seg_mode=MULTICLASS_MODE, debug=False):
        """
        initial all the characteristic

        Inputs:
        -params: configuration parameters
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize

        Returns:
        None
        """
        self.is_train = is_train
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(params.dataset['dataroot'])
        self.seg_list = params.seg_list
        if is_train:
            self.indicator = params.dataset['train_set']
        else:
            self.indicator = params.dataset['test_set']
        self.img_root = img_root / self.indicator
        self.label_root = params.dataset['labelroot']
        with open(self.label_root, 'r') as f:
            self.label_list = json.load(f)
        if debug:
            self.label_list = self.label_list[:10]
        self.albumentations_transform = A.Compose([
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.01),
            A.RandomGamma(p=0.01),
            A.ImageCompression(quality_lower=75, p=0.01)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
            additional_targets={'mask0': 'mask'})
        

        self.shapes = np.array(params.dataset['org_img_size'])
        self.obj_combine = params.obj_combine
        self.obj_list = params.obj_list
        self.dataset = params.dataset
        self.traffic_light_color = params.traffic_light_color
        self.mosaic_border = [-1 * self.inputsize[1] // 2, -1 * self.inputsize[0] // 2]
        self.seg_mode = seg_mode
        self.db = self._get_db()

    def _get_db(self):
        """
        TODO: add docs
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        for label in tqdm(self.label_list, ascii=True):
            if label['split'] != self.indicator: continue
            image_path = str(self.img_root / label['file_upload'])
            gt = []
            # this is assuming that the dataset has same image size, which is convenient
            seg_label = OrderedDict()
            for seg_class in self.seg_list:
                    seg_label[seg_class] = np.zeros((label['annotations'][0]['result'][0]['original_height'], label['annotations'][0]['result'][0]['original_width']), dtype=np.uint8)
            for result in label['annotations'][0]['result']:
                if result['type'] == 'polygonlabels':
                    mask = np.zeros((result['original_height'], result['original_width']), dtype=np.uint8)
                    # print(result['value']['points'])
                    vertices = np.array(result['value']['points'])
                    # print(vertices)
                    # exit()
                    rr, cc = polygon(vertices[:,1], vertices[:,0], mask.shape)
                    mask[rr, cc] = 255
                    # torchshow.save(mask, '_mask.png')
                    category = result['value']['polygonlabels'][0].lower()
                    if category in self.seg_list:
                        seg_label[seg_class] |= mask

                elif result['type'] == 'rectanglelabels':
                    w = float(round(result['value']['width']))
                    h = float(round(result['value']['height']))
                    x = float(round(result['value']['x'] + w / 2))
                    y = float(round(result['value']['y'] + h / 2))
                    try:
                        category = result['value']['rectanglelabels'][0].lower()
                    except:
                        print(result)
                    cls_id = self.obj_list.index(category)
                    gt.append([cls_id, x, y, w, h])
            # print(gt)
            # torchshow.save(seg_label['road'], path='_roadfromb4.png')

            gt = np.array(gt)

            rec = {
                # 'image': img,
                'image': image_path,
                'label': gt,
            }
            # Since seg_path is a dynamic dict
            # rec = {**rec, **seg_path}
            rec = {**rec, **seg_label}

            # img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_UNCHANGED)
            # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # for label in gt:
            #     # print(label[1])
            #     x1 = label[1] - label[3] / 2
            #     x1 *= 1280
            #     x1 = int(x1)
            #     # print(x1)
            #     x2 = label[1] + label[3] / 2
            #     x2 *= 1280
            #     x2 = int(x2)
            #     y1 = label[2] - label[4] / 2
            #     y1 *= 720
            #     y1 = int(y1)
            #     y2 = label[2] + label[4] / 2
            #     y2 *= 720
            #     y2 = int(y2)
            #     img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.imwrite('gt/{}'.format(image_path.split('/')[-1]), img)

            gt_db.append(rec)
        print('database build finish')
        return gt_db


    def load_image(self, index):
        data = self.db[index]
        det_label = data["label"]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = data["image"]
        seg_label = OrderedDict()
        for seg_class in self.seg_list:
            # seg_label[seg_class] = cv2.imread(data[seg_class], 0)
            seg_label[seg_class] = data[seg_class]

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_sizeWW
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            for seg_class in self.seg_list:
                seg_label[seg_class] = cv2.resize(seg_label[seg_class], (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
        # print(h, w)
        # exit()
    
        labels = []
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = (det_label[:, 1] - det_label[:, 3] / 2) / w0 * w # pad width
            labels[:, 2] = (det_label[:, 2] - det_label[:, 4] / 2) / h0 * h # pad height
            labels[:, 3] = (det_label[:, 1] + det_label[:, 3] / 2) / w0 * w
            labels[:, 4] = (det_label[:, 2] + det_label[:, 4] / 2) / h0 * h

#         img_clone = img.copy()
#         for anno in labels:
#           x1,y1,x2,y2 = [int(x) for x in anno[1:5]]
#           img_clone = cv2.rectangle(img_clone, (x1,y1), (x2,y2), (255,0,0), 1)
#         cv2.imwrite("label-{}.jpg".format(index), img_clone)

        for seg_class in seg_label:
            _, seg_label[seg_class] = cv2.threshold(seg_label[seg_class], 0, 255, cv2.THRESH_BINARY)
        # np.savetxt('seglabelroad_before', seg_label['road'])
    
        return img, labels, seg_label, (h0, w0), (h,w), None # data['image']


    @staticmethod
    def collate_fn(batch):
        img, paths, shapes, labels_app, segmentation = zip(*batch)
        # filenames = [file.split('/')[-1] for file in paths]
        # print(len(labels_app))
        max_num_annots = max(label.size(0) for label in labels_app)

        if max_num_annots > 0:
            annot_padded = torch.ones((len(labels_app), max_num_annots, 5)) * -1
            for idx, label in enumerate(labels_app):
                if label.size(0) > 0:
                    annot_padded[idx, :label.size(0), :] = label
        else:
            annot_padded = torch.ones((len(labels_app), 1, 5)) * -1

        # print("ABC", seg1.size())
        return {'img': torch.stack(img, 0), 'annot': annot_padded, 'segmentation': torch.stack(segmentation, 0),
                'filenames': None, 'shapes': shapes}
