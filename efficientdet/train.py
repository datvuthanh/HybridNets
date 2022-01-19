# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

import argparse
import datetime
import os
import traceback
import gc

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
from tqdm.autonotebook import tqdm

from val import val
from backbone import EfficientDetBackbone
from backbone_old import EfficientDetBackbone as EfficientDetBackboneOld
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string, \
    save_checkpoint
from efficientdet.bdd import BddDataset
from efficientdet.AutoDriveDataset import AutoDriveDataset
from efficientdet.yolop_cfg import update_config
from efficientdet.yolop_cfg import _C as cfg
from efficientdet.yolop_utils import DataLoaderX
from utils.dice_loss import DiceLoss
from utils.focal_loss import FocalLoss as FocalLossSeg
from utils.tversky_loss import TverskyLoss
from utils.dice_loss_old import DiceLoss as DiceLossOld
from utils.lovasz_loss import LovaszLoss


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='checkpoints/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='checkpoints/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')
    parser.add_argument('--is_transfer', type=boolean_string, default=True,
                        help='transfer learning from pretrained effdet')
    parser.add_argument('--cal_map', type=boolean_string, default=True,
                        help='calculate map in validation')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        # self.seg_criterion1 = DiceLoss(mode='binary', from_logits=False)
        # self.seg_criterion1 = DiceLossOld()
        self.seg_criterion1 = TverskyLoss(mode='multilabel', alpha=0.7, beta=0.3, gamma=4.0/3, from_logits=False)
        # self.seg_criterion1 = LovaszLoss(mode='binary')
        self.seg_criterion2 = FocalLossSeg(mode='multilabel', alpha=0.25)
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations,seg_annot, obj_list=None):
        _, regression, classification, anchors, segmentation = self.model(imgs)

        if self.debug:
            cls_loss, reg_loss= self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
            dice_loss = self.seg_criterion1(segmentation, seg_annot)
            tversky_loss = self.seg_criterion1(segmentation, seg_annot)
            focal_loss = self.seg_criterion2(segmentation, seg_annot)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
            # Calculate segmentation loss
            # dice_loss = self.seg_criterion1(segmentation, seg_annot)
            tversky_loss = self.seg_criterion1(segmentation, seg_annot)
            focal_loss = self.seg_criterion2(segmentation, seg_annot)
            # lovasz_loss = self.seg_criterion1(segmentation, seg_annot)

            # Visualization
            # seg_0 = seg_annot[0]
            # # print('bbb', seg_0.shape)
            # seg_0 = torch.argmax(seg_0, dim = 0)
            # # print('before', seg_0.shape)
            # seg_0 = seg_0.cpu().numpy()
            #     #.transpose(1, 2, 0)
            # print(seg_0.shape)
            #
            # anh = np.zeros((384,640,3))
            #
            # anh[seg_0 == 0] = (255,0,0)
            # anh[seg_0 == 1] = (0,255,0)
            # anh[seg_0 == 2] = (0,0,255)
            #
            # anh = np.uint8(anh)
            #
            # cv2.imwrite('anh.jpg',anh)

        seg_loss = tversky_loss + 1 * focal_loss
        # seg_loss = lovasz_loss
        # print("DICE", dice_loss)
        # print("FOCAL", focal_loss)

        return cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation


def train(opt):
    params = Params(f'projects/{opt.project}.yml')
    update_config(cfg, opt)

    # use all in environment
    params.num_gpus = torch.cuda.device_count()
    print("Using {} GPUS".format(params.num_gpus))

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    train_dataset = BddDataset(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])
    )

    training_generator = DataLoaderX(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=AutoDriveDataset.collate_fn
    )

    valid_dataset = BddDataset(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])
    )

    val_generator = DataLoaderX(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=AutoDriveDataset.collate_fn
    )

    pretrainedmodels = EfficientDetBackboneOld(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales), seg_classes = len(params.seg_list))

    # load last weights
    ckpt = {}
    # last_step = None
    if opt.load_weights is not None and opt.is_transfer:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        # try:
        #     last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        # except:
        #     last_step = 0

        try:
            ret = pretrainedmodels.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}')

        print('[Info] Load a part of weights from pretrained models')

        params1 = pretrainedmodels.state_dict()
        params2 = model.state_dict()

        dict_params1 = dict(params1)
        dict_params2 = dict(params2)

        pretrained_dict = {k: v for k, v in dict_params1.items() if k in dict_params2}
        dict_params2.update(pretrained_dict)
        model.load_state_dict(dict_params2)
    elif opt.load_weights is not None and not opt.is_transfer:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        # try:
        #     last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        # except:
        #     last_step = 0

        try:
            ckpt = torch.load(weights_path)
            model.load_state_dict(ckpt.get('model', ckpt), strict=False)

        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
    else:
        print('[Info] initializing weights...')
        init_weights(model)

    print('[Info] Successfully!!!')

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # wrap the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)
    # print(ckpt)
    if opt.load_weights is not None and ckpt.get('optimizer', None):
        optimizer.load_state_dict(ckpt['optimizer'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    last_step = ckpt['step'] if  opt.load_weights is not None and ckpt.get('step', None) else 0
    best_fitness = ckpt['best_fitness'] if opt.load_weights is not None and ckpt.get('best_fitness', None) else 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']
                    seg_annot = data['segmentation']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()
                        seg_annot = seg_annot.cuda().long()

                    optimizer.zero_grad()
                    cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation= model(imgs, annot, seg_annot,obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()
                    seg_loss = seg_loss.mean()

                    loss = cls_loss + reg_loss + seg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Seg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), seg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)
                    writer.add_scalars('Segmentation_loss', {'train': seg_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, opt.saved_path, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                best_fitness, best_loss, best_epoch = val(model, optimizer, val_generator, params, opt, writer, epoch,
                                                          step, best_fitness, best_loss, best_epoch)
    except KeyboardInterrupt:
        # save_checkpoint(model, opt.saved_path, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        writer.close()
    writer.close()


if __name__ == '__main__':
    opt = get_args()
    train(opt)
