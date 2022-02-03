import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset
from .yolop_utils import letterbox, augment_hsv, random_perspective


class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """

    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize

        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.lane_root = lane_root / indicator
        # self.label_list = self.label_root.iterdir()
        self.mask_list = self.mask_root.iterdir()

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
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

        if self.cfg.num_seg_class == 3:
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
            # augmentation
            combination = (img, seg_label, lane_label)
            (img, seg_label, lane_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR
            )
#             print(labels.shape)
            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)
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
                    labels[:, 2] = 1 - labels[:, 2]

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
        union = seg1 | lane1
        # print(union.shape)
        background = 255 - union

        # print(img.shape)
        # print(lane1.shape)
        # img_copy = img.copy()
        # img_copy[lane1 == 255] = (0, 255, 0)
        # cv2.imwrite('seg_gt/' + data['image'].split('/')[-1], img_copy)
        # cv2.imwrite('background.jpg', background)
        # cv2.imwrite('lane1.jpg',lane1)
        # cv2.imwrite('seg1.jpg',seg1)

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
        db_selected = ...
        return db_selected

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
