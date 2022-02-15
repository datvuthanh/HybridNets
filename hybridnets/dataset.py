import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset
from utils.utils import letterbox, augment_hsv, random_perspective
from tqdm.autonotebook import tqdm
import json
import albumentations as A


class BddDataset(Dataset):
    def __init__(self, params, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -params: configuration parameters
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize

        Returns:
        None
        """
        self.single_cls = True  # just detect vehicle
        self.is_train = is_train
        self.params = params
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(params.dataset['dataroot'])
        label_root = Path(params.dataset['labelroot'])
        mask_root = Path(params.dataset['maskroot'])
        lane_root = Path(params.dataset['laneroot'])
        if is_train:
            indicator = params.dataset['train_set']
        else:
            indicator = params.dataset['test_set']
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.lane_root = lane_root / indicator
        # self.label_list = self.label_root.iterdir()
        self.mask_list = self.mask_root.iterdir()
        self.data_format = params.dataset['data_format']
        self.scale_factor = params.dataset['scale_factor']
        self.rotation_factor = params.dataset['rot_factor']
        self.flip = params.dataset['flip']
        self.color_rgb = params.dataset['color_rgb']
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

        # bdd_labels = {
        # 'unlabeled':0, 'dynamic': 1, 'ego vehicle': 2, 'ground': 3,
        # 'static': 4, 'parking': 5, 'rail track': 6, 'road': 7,
        # 'sidewalk': 8, 'bridge': 9, 'building': 10, 'fence': 11,
        # 'garage': 12, 'guard rail': 13, 'tunnel': 14, 'wall': 15,
        # 'banner': 16, 'billboard': 17, 'lane divider': 18,'parking sign': 19,
        # 'pole': 20, 'polegroup': 21, 'street light': 22, 'traffic cone': 23,
        # 'traffic device': 24, 'traffic light': 25, 'traffic sign': 26, 'traffic sign frame': 27,
        # 'terrain': 28, 'vegetation': 29, 'sky': 30, 'person': 31,
        # 'rider': 32, 'bicycle': 33, 'bus': 34, 'car': 35,
        # 'caravan': 36, 'motorcycle': 37, 'trailer': 38, 'train': 39,
        # 'truck': 40
        # }
        self.id_dict = {'person': 0, 'rider': 1, 'car': 2, 'bus': 3, 'truck': 4,
                'bike': 5, 'motor': 6, 'tl_green': 7, 'tl_red': 8,
                'tl_yellow': 9, 'tl_none': 10, 'traffic sign': 11, 'train': 12}
        self.id_dict_single = {'car': 0, 'bus': 1, 'truck': 2, 'train': 3}
        # id_dict = {'car': 0, 'bus': 1, 'truck': 2}

        self.shapes = np.array(params.dataset['org_img_size'])
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
        for mask in tqdm(list(self.mask_list)):
            mask_path = str(mask)
            label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".json")
            image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
            lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
            with open(label_path, 'r') as f:
                label = json.load(f)
            data = label['frames'][0]['objects']
            data = self.select_data(data)
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['category']
                if category == "traffic light":
                    color = obj['attributes']['trafficLightColor']
                    category = "tl_" + color
                if category in self.id_dict.keys():
                    x1 = float(obj['box2d']['x1'])
                    y1 = float(obj['box2d']['y1'])
                    x2 = float(obj['box2d']['x2'])
                    y2 = float(obj['box2d']['y2'])
                    cls_id = self.id_dict[category]
                    if self.single_cls:
                        cls_id = 0
                    gt[idx][0] = cls_id
                    box = self.convert((width, height), (x1, x2, y1, y2))
                    gt[idx][1:] = list(box)

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path
            }]

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

            gt_db += rec
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
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.params.num_seg_class == 3:
            seg_label = cv2.imread(data["mask"])
        else:
            seg_label = cv2.imread(data["mask"], 0)
        lane_label = cv2.imread(data["lane"], 0)

        # print(lane_label.shape)
        # print(seg_label.shape)
        # print(lane_label.shape)
        # print(seg_label.shape)
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        (img, seg_label, lane_label), ratio, pad = letterbox((img, seg_label, lane_label), resized_shape, auto=True,
                                                             scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # ratio = (w / w0, h / h0)
        # print(resized_shape)

        det_label = data["label"]
        # print(det_label)

        labels = []
        labels_app = np.array([])

        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]

        # print(labels[:, 1:4])
        if self.is_train:
            # albumentations
            try:
                new = self.albumentations_transform(image=img, mask=seg_label, mask0=lane_label,
                                                    bboxes=labels[:, 1:] if len(labels) else labels,
                                                    class_labels=labels[:, 0] if len(labels) else labels)
                img = new['image']
                labels = np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]) if len(labels) else labels
                seg_label = new['mask']
                lane_label = new['mask0']
            except ValueError:  # bbox have width or height == 0
                pass

            # augmentation
            combination = (img, seg_label, lane_label)
            (img, seg_label, lane_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=self.params.dataset['rot_factor'],
                translate=self.params.dataset['translate'],
                scale=self.params.dataset['scale_factor'],
                shear=self.params.dataset['shear']
            )
#             print(labels.shape)
            augment_hsv(img, hgain=self.params.dataset['hsv_h'], sgain=self.params.dataset['hsv_s'], vgain=self.params.dataset['hsv_v'])
#             img, seg_label, labels = cutout(combination=combination, labels=labels)

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
                seg_label = np.fliplr(seg_label)
                lane_label = np.fliplr(lane_label)
                
#                 cv2.imwrite('img0.jpg',img)
#                 cv2.imwrite('img1.jpg',seg_label)
#                 cv2.imwrite('img2.jpg',lane_label)
                
#                 exit()

            # print(labels)

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                seg_label = np.filpud(seg_label)
                lane_label = np.filpud(lane_label)
                if len(labels):
                    rows, cols, channels = img.shape

                    y1 = labels[:, 2].copy()
                    y2 = labels[:, 4].copy()

                    y_tmp = y1.copy()

                    labels[:, 2] = rows - y2
                    labels[:, 4] = rows - y_tmp

        # for anno in labels:
        #   x1, y1, x2, y2 = [int(x) for x in anno[1:5]]
        #   print(x1,y1,x2,y2)
        #   cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
        # cv2.imwrite(data["image"].split("/")[-1], img)

        if len(labels):
            labels_app = np.zeros((len(labels), 5))
            labels_app[:, 0:4] = labels[:, 1:5]
            labels_app[:, 4] = labels[:, 0]

        img = np.ascontiguousarray(img)

        _, seg1 = cv2.threshold(seg_label, 1, 255, cv2.THRESH_BINARY)
        _, lane1 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY)
        # prefer lane
        seg1 = seg1 - (seg1 & lane1)

        union = seg1 | lane1
        # print(union.shape)
        background = 255 - union

        #         print(img.shape)
        #         print(lane1.shape)
        #         img_copy = img.copy()
        #         img_copy[lane1 == 255] = (0, 255, 0)
        #         cv2.imwrite('seg_gt/' + data['image'].split('/')[-1], img_copy)
        #         cv2.imwrite('background.jpg', background)
        #         cv2.imwrite('{}.jpg'.format(data['image'].split('/')[-1]), img)
        #         cv2.imwrite('{}-lane.jpg'.format(data['image'].split('/')[-1]),lane1)
        #         cv2.imwrite('{}-seg.jpg'.format(data['image'].split('/')[-1]),seg1)

        seg1 = self.Tensor(seg1)
        lane1 = self.Tensor(lane1)
        background = self.Tensor(background)

        segmentation = torch.cat([background, seg1, lane1], dim=0)
        # print(segmentation.size())
        # print(seg1.shape)

        # for anno in labels_app:
        #   x1, y1, x2, y2 = [int(x) for x in anno[anno != -1][:4]]
        #   cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)
        # cv2.imwrite(data["image"].split("/")[-1], img)

        img = self.transform(img)

        return img, data["image"], shapes, torch.from_numpy(labels_app), segmentation

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
                if self.single_cls:
                    if obj['category'] in self.id_dict_single.keys():
                        remain.append(obj)
                else:
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
