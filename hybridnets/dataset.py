import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
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


class BddDataset(Dataset):
    def __init__(self, params, is_train, inputsize=[640, 384], transform=None, use_mosaic=False, seg_mode=MULTICLASS_MODE):
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
        label_root = Path(params.dataset['labelroot'])
        seg_root = params.dataset['segroot']
        self.seg_list = params.seg_list
        if is_train:
            indicator = params.dataset['train_set']
        else:
            indicator = params.dataset['test_set']
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.label_list = list(self.label_root.iterdir())
        self.seg_root = []
        for root in seg_root:
            self.seg_root.append(Path(root) / indicator)
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
        self.use_mosaic = use_mosaic
        self.mosaic_border = [-1 * self.inputsize[1] // 2, -1 * self.inputsize[0] // 2]
        self.seg_mode = seg_mode
        self.db = self._get_db()

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        for label in tqdm(self.label_list, ascii=True):
            label_path = str(label)
            image_path = label_path.replace(str(self.label_root), str(self.img_root)).replace(".json", ".jpg")
            seg_path = {}
            for i in range(len(self.seg_list)):
                seg_path[self.seg_list[i]] = label_path.replace(str(self.label_root), str(self.seg_root[i])).replace(".json", ".png")
            with open(label_path, 'r') as f:
                label = json.load(f)
            data = label['frames'][0]['objects']
            data = self.select_data(data)
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['category']
                x1 = float(obj['box2d']['x1'])
                y1 = float(obj['box2d']['y1'])
                x2 = float(obj['box2d']['x2'])
                y2 = float(obj['box2d']['y2'])
                if len(self.obj_combine):  # multiple classes into 1 class
                    cls_id = 0
                else:
                    cls_id = self.obj_list.index(category)
                gt[idx][0] = cls_id
                box = self.convert((width, height), (x1, x2, y1, y2))
                gt[idx][1:] = list(box)

            rec = {
                'image': image_path,
                'label': gt,
            }
            # Since seg_path is a dynamic dict
            rec = {**rec, **seg_path}

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


    def evaluate(self, params, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError

    def __len__(self, ):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def load_image(self, index):
        data = self.db[index]
        det_label = data["label"]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg_label = OrderedDict()
        for seg_class in self.seg_list:
            seg_label[seg_class] = cv2.imread(data[seg_class], 0)

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
    
        labels = []
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = (det_label[:, 1] - det_label[:, 3] / 2) * w # pad width
            labels[:, 2] = (det_label[:, 2] - det_label[:, 4] / 2) * h # pad height
            labels[:, 3] = (det_label[:, 1] + det_label[:, 3] / 2) * w
            labels[:, 4] = (det_label[:, 2] + det_label[:, 4] / 2) * h

#         img_clone = img.copy()
#         for anno in labels:
#           x1,y1,x2,y2 = [int(x) for x in anno[1:5]]
#           img_clone = cv2.rectangle(img_clone, (x1,y1), (x2,y2), (255,0,0), 1)
#         cv2.imwrite("label-{}.jpg".format(index), img_clone)
    
        return img, labels, seg_label, (h0, w0), (h,w), data['image']

    def load_mosaic(self, index):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        w_mosaic, h_mosaic = self.inputsize
        yc = int(random.uniform(-self.mosaic_border[0], 2 * h_mosaic + self.mosaic_border[0]))
        xc = int(random.uniform(-self.mosaic_border[1], 2 * w_mosaic + self.mosaic_border[1]))
        
        indices = range(len(self.db))
        indices = [index] + random.choices(indices, k=3)  # 3 additional iWmage indices
                        
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, labels, seg_label, (h0,w0), (h, w), path = self.load_image(index)
                        
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((h_mosaic * 2, w_mosaic * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w_mosaic * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h_mosaic * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w_mosaic * 2), min(h_mosaic * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            
            if len(labels):
                labels[:, 1] += padw
                labels[:, 2] += padh
                labels[:, 3] += padw
                labels[:, 4] += padh
            
                labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        
        new = labels4.copy()
        new[:, 1:] = np.clip(new[:, 1:], 0, 2*w_mosaic)
        new[:, 2:5:2] = np.clip(new[:, 2:5:2], 0, 2*h_mosaic)

        # filter candidates
        i = box_candidates(box1=labels4[:,1:5].T, box2=new[:,1:5].T)
        labels4 = labels4[i]
        labels4[:] = new[i] 

        return img4, labels4, seg_label, (h0, w0), (h, w), path

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        if self.is_train:
            if self.use_mosaic:
            # TODO: this doubles training time with inherent stuttering in tqdm, prob cpu or io bottleneck, does prefetch_generator work with ddp?
            # TODO: updated, mosaic is inherently slow, maybe cache the images in RAM? maybe it was IO bottleneck of reading 4 images everytime? time it
            # honestly, mosaic is not for road and lane segmentation anyway
            # you cant expect road and lane to be split up in 4 separate corners in an image, do you?
            # only use mosaic with freeze_seg :)
                img, labels, seg_label, (h0, w0), (h, w), path = self.load_mosaic(idx)
                # mixup is double mosaic, really slow
                # if random.random() < 0.2:
                #     img2, labels2, _, (_, _), (_, _), _ = self.load_mosaic(random.randint(0, len(self.db) - 1))
                #     img, labels = mixup(img, labels, img2, labels2)
            # albumentations
            else:
                img, labels, seg_label, (h0, w0), (h, w), path = self.load_image(idx)
            # TODO: multi-class seg with albumentations
            # try:
            #     new = self.albumentations_transform(image=img, mask=seg_label, mask0=lane_label,
            #                                         bboxes=labels[:, 1:] if len(labels) else labels,
            #                                         class_labels=labels[:, 0] if len(labels) else labels)
            #     img = new['image']
            #     labels = np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]) if len(labels) else labels
            #     seg_label = new['mask']
            #     lane_label = new['mask0']
            # except ValueError:  # bbox have width or height == 0
            #     pass

            # augmentation
            combination = (img, seg_label)
            (img, seg_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=self.dataset['rot_factor'],
                translate=self.dataset['translate'],
                scale=self.dataset['scale_factor'],
                shear=self.dataset['shear'],
                border=self.mosaic_border if self.use_mosaic else (0, 0)
            )
            augment_hsv(img, hgain=self.dataset['hsv_h'], sgain=self.dataset['hsv_s'], vgain=self.dataset['hsv_v'])

            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = img[:, ::-1, :]

                if len(labels):
                    rows, cols, channels = img.shape

                    x1 = labels[:, 1].copy()
                    x2 = labels[:, 3].copy()

                    x_tmp = x1.copy()

                    labels[:, 1] = cols - x2
                    labels[:, 3] = cols - x_tmp
                
                # Segmentation
                for seg_class in seg_label:
                    seg_label[seg_class] = np.fliplr(seg_label[seg_class])
                    # lane_label = np.fliplr(lane_label)

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)

                if len(labels):
                    rows, cols, channels = img.shape

                    y1 = labels[:, 2].copy()
                    y2 = labels[:, 4].copy()

                    y_tmp = y1.copy()

                    labels[:, 2] = rows - y2
                    labels[:, 4] = rows - y_tmp

                # Segmentation
                for seg_class in seg_label:
                    seg_label[seg_class] = np.flipud(seg_label[seg_class])
                
        else:
            img, labels, seg_label, (h0, w0), (h, w), path = self.load_image(idx)

        # for anno in labels:
        #   x1, y1, x2, y2 = [int(x) for x in anno[1:5]]
        #   print(x1,y1,x2,y2)
        #   cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
        # cv2.imwrite(data["image"].split("/")[-1], img)

        (img, seg_label), ratio, pad = letterbox((img, seg_label), (self.inputsize[1], self.inputsize[0]), auto=True,
                                                             scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling  
        
        labels_app = np.array([])
        if len(labels):
            # update labels after letterbox
            labels[:, 1] = ratio[0] * labels[:, 1] + pad[0]
            labels[:, 2] = ratio[1] * labels[:, 2] + pad[1]
            labels[:, 3] = ratio[0] * labels[:, 3] + pad[0]
            labels[:, 4] = ratio[1] * labels[:, 4] + pad[1]     

            labels_app = np.zeros((len(labels), 5))
            labels_app[:, 0:4] = labels[:, 1:5]
            labels_app[:, 4] = labels[:, 0]

        img = np.ascontiguousarray(img)

        for seg_class in seg_label:
            _, seg_label[seg_class] = cv2.threshold(seg_label[seg_class], 1, 255, cv2.THRESH_BINARY)
        
        if self.seg_mode == BINARY_MODE:
            for seg_class in seg_label:
                # technically, the for-loop only goes once
                segmentation = self.Tensor(seg_label[seg_class])
            
            # [1, H, W]
            # road [0, 0, 0, 0]
            #      [0, 1, 1, 0]
            #      [0, 1, 1, 0]
            #      [1, 1, 1, 1]

        elif self.seg_mode == MULTICLASS_MODE:
            # special treatment for lane-line of bdd100k for our dataset
            # since we increase lane-line from 2 to 8 pixels, we must take care of the overlap to other segmentation classes
            # e.g.: a pixel belongs to both road and lane-line, then we must prefer lane, or metrics would be wrong
            if 'lane' in seg_label:
                for seg_class in seg_label:
                    if seg_class != 'lane': seg_label[seg_class] -= seg_label['lane']

            segmentation = np.zeros(img.shape[:2], dtype=np.uint8)
            segmentation = self.Tensor(segmentation)
            segmentation.squeeze_(0)
            for seg_index, seg_class in enumerate(seg_label.values()):
                segmentation[seg_class == 255] = seg_index + 1
            
            # [H, W]
            # background = 0, road = 1, lane = 2
            # [0, 0, 0, 0]
            # [2, 1, 1, 2]
            # [2, 1, 1, 2]
            # [1, 1, 1, 1]

        else:  # multi-label
            union = np.zeros(img.shape[:2], dtype=np.uint8)
            for seg_class in seg_label:
                union |= seg_label[seg_class]
            background = 255 - union

            for seg_class in seg_label:
                seg_label[seg_class] = self.Tensor(seg_label[seg_class])
            background = self.Tensor(background)
            segmentation = torch.cat([background, *seg_label.values()], dim=0)

            # [C, H, W]
            # background [1, 1, 1, 1] road [0, 0, 0, 0]   lane [0, 0, 0, 0]
            #            [0, 0, 0, 0]      [0, 1, 1, 0]        [1, 0, 0, 1]
            #            [0, 0, 0, 0]      [0, 1, 1, 0]        [1, 0, 0, 1]
            #            [0, 0, 0, 0]      [1, 1, 1, 1]        [1, 0, 0, 1]

        # print(img.shape)
        # print(lane1.shape)
        # img_copy = img.copy()
        # img_copy[seg_label['road'] == 255] = (0, 255, 0)
        # cv2.imwrite('_copy.jpg', img_copy)
        # cv2.imwrite('_seg.jpg', seg_label['road'])
        # cv2.imwrite('_background.jpg', background)
        # cv2.imwrite('{}-seg.jpg'.format(data['image'].split('/')[-1]),seg1)

        # for anno in labels_app:
        #   x1, y1, x2, y2 = [int(x) for x in anno[anno != -1][:4]]
        #   cv2.rectangle(img_copy, (x1,y1), (x2,y2), (0,0,255), 1)
        # cv2.imwrite('_box.jpg', img_copy)

        img = self.transform(img)

        return img, path, shapes, torch.from_numpy(labels_app), segmentation.long()

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        remain = []
        for obj in db:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if self.traffic_light_color and obj['category'] == "traffic light":
                    color = obj['attributes']['trafficLightColor']
                    obj['category'] = "tl_" + color
                if obj['category'] in self.obj_list:  # multi-class
                    remain.append(obj)
                elif len(self.obj_list) == 1 and obj['category'] in self.obj_combine:
                    remain.append(obj)
        return remain

    def convert(self, size, box):
            dw = 1. / (size[0])
            dh = 1. / (size[1])
            x = (box[0] + box[1]) / 2.0
            y = (box[2] + box[3]) / 2.0
            w = box[1] - box[0]
            h = box[3] - box[2]
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            return x, y, w, h
        
    @staticmethod
    def collate_fn(batch):
        img, paths, shapes, labels_app, segmentation = zip(*batch)
        filenames = [file.split('/')[-1] for file in paths]
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
                'filenames': filenames, 'shapes': shapes}
