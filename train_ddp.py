import argparse
import datetime
import os
import traceback

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
from tqdm.autonotebook import tqdm

from val_ddp import val
from backbone import HybridNetsBackbone
from hybridnets.loss import FocalLoss
from utils.utils import get_last_weights, init_weights, boolean_string, save_checkpoint, Params
from hybridnets.dataset import BddDataset
from hybridnets.loss import FocalLossSeg, TverskyLoss
from hybridnets.autoanchor import run_anchor
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                            'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='Num_workers of dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=12, help='Number of images per batch among all devices')
    parser.add_argument('--freeze_backbone', type=boolean_string, default=False,
                        help='Freeze encoder and neck (effnet and bifpn)')
    parser.add_argument('--freeze_det', type=boolean_string, default=False,
                        help='Freeze detection head')
    parser.add_argument('--freeze_seg', type=boolean_string, default=False,
                        help='Freeze segmentation head')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='Select optimizer for training, '
                                                                   'suggest using \'adamw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which '
                             'training will be stopped. Set to 0 to disable this technique')
    parser.add_argument('--data_path', type=str, default='datasets/', help='The root folder of dataset')
    parser.add_argument('--log_path', type=str, default='checkpoints/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='Whether to load weights from a checkpoint, set None to initialize,'
                             'set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='checkpoints/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='Whether visualize the predicted boxes of training, '
                             'the output images will be in test/')
    parser.add_argument('--cal_map', type=boolean_string, default=True,
                        help='Calculate mAP in validation')
    parser.add_argument('-v', '--verbose', type=boolean_string, default=True,
                        help='Whether to print results per class when valing')
    parser.add_argument('--plots', type=boolean_string, default=True,
                        help='Whether to plot confusion matrix when valing')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to be used (0 to use CPU)')
    parser.add_argument('--mosaic', type=boolean_string, default=False,
                        help='Use mosaic augmentation, '
                             'recommended when training object detection only.')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.seg_criterion1 = TverskyLoss(mode='multilabel', alpha=0.7, beta=0.3, gamma=4.0 / 3, from_logits=False)
        self.seg_criterion2 = FocalLossSeg(mode='multilabel', alpha=0.25)
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, seg_annot, obj_list=None):
        _, regression, classification, anchors, segmentation = self.model(imgs)

        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
            tversky_loss = self.seg_criterion1(segmentation, seg_annot)
            focal_loss = self.seg_criterion2(segmentation, seg_annot)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
            tversky_loss = self.seg_criterion1(segmentation, seg_annot)
            focal_loss = self.seg_criterion2(segmentation, seg_annot)

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
        # print("TVERSKY", tversky_loss)
        # print("FOCAL", focal_loss)

        return cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation


def train(rank, opt):
    print(2)
    params = Params(f'projects/{opt.project}.yml')

    torch.cuda.manual_seed(69)
    torch.manual_seed(69)

    setup(rank, opt.num_gpus)

    train_dataloader, val_dataloader = prepare(rank, params, opt)

    model = HybridNetsBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                               ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
                               seg_classes=len(params.seg_list), backbone_name=opt.backbone)

    # load last weights
    ckpt = {}
    # last_step = None
    if opt.load_weights:
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

    if opt.freeze_backbone:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            if classname in ['EfficientNetEncoder', 'BiFPN']:  # replace backbone classname when using another backbone
                print("[Info] freezing {}".format(classname))
                for param in m.parameters():
                    param.requires_grad = False
        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    if opt.freeze_det:
        def freeze_det(m):
            classname = m.__class__.__name__
            if classname in ['Regressor', 'Classifier', 'Anchors']:
                print("[Info] freezing {}".format(classname))
                for param in m.parameters():
                    param.requires_grad = False
        model.apply(freeze_det)
        print('[Info] freezed detection head')

    if opt.freeze_seg:
        def freeze_seg(m):
            classname = m.__class__.__name__
            if classname in ['BiFPNDecoder', 'SegmentationHead']:
                print("[Info] freezing {}".format(classname))
                for param in m.parameters():
                    param.requires_grad = False
        model.apply(freeze_seg)
        print('[Info] freezed segmentation head')

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # wrap the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if opt.optim == 'adamw':
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=opt.lr
        )
    else:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.SGD,
            lr=opt.lr,
            momentum=0.9,
            nesterov=True
        )
    # print(ckpt)
    # if opt.load_weights is not None and ckpt.get('optimizer', None):
    #     optimizer.load_state_dict(ckpt['optimizer'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    last_step = ckpt['step'] if opt.load_weights is not None and ckpt.get('step', None) else 0
    best_fitness = ckpt['best_fitness'] if opt.load_weights is not None and ckpt.get('best_fitness', None) else 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(train_dataloader)
    torch.cuda.set_device(rank)
    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            train_dataloader.sampler.set_epoch(epoch)
            progress_bar = tqdm(train_dataloader)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch and rank == 0:
                    progress_bar.update()
                    continue
                try:
                    # print("WTF")
                    imgs = data['img'].to(rank)
                    annot = data['annot'].to(rank)
                    seg_annot = data['segmentation'].to(rank)

                    optimizer.zero_grad()
                    cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot,
                                                                                                            seg_annot,
                                                                                                            obj_list=params.obj_list)
                    cls_loss = cls_loss.mean() # if not opt.freeze_det else 0
                    reg_loss = reg_loss.mean() # if not opt.freeze_det else 0
                    seg_loss = seg_loss.mean() # if not opt.freeze_seg else 0

                    loss = cls_loss + reg_loss + seg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))
                    
                    # print(3)
                    dist.reduce(loss, 0, op=dist.ReduceOp.AVG)
                    # print(4)
                    dist.reduce(cls_loss, 0, op=dist.ReduceOp.AVG)
                    # print(5)

                    dist.reduce(reg_loss, 0, op=dist.ReduceOp.AVG)
                    dist.reduce(seg_loss, 0, op=dist.ReduceOp.AVG)
                    # print(6)



                    if rank == 0:
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

                    if step % opt.save_interval == 0 and step > 0 and rank == 0:
                        save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            epoch_loss_tensor = torch.tensor(np.mean(epoch_loss), device=rank)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.AVG)
            scheduler.step(epoch_loss_tensor.item())

            if epoch % opt.val_interval == 0:
                best_fitness, best_loss, best_epoch = val(model, rank, optimizer, val_dataloader, params, opt, writer, epoch,
                                                          step, best_fitness, best_loss, best_epoch)
    except KeyboardInterrupt:
        if rank == 0:
            save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth')
    finally:
        writer.close()
        dist.destroy_process_group()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23456'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(rank, params, opt):
    train_dataset = BddDataset(
        params=params,
        is_train=True,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        use_mosaic=opt.mosaic
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=opt.num_gpus, rank=rank, shuffle=False, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, pin_memory=params.pin_memory, num_workers=opt.num_workers,
                                    drop_last=True, shuffle=False, sampler=train_sampler, collate_fn=BddDataset.collate_fn)
    
    val_dataset = BddDataset(
        params=params,
        is_train=False,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=opt.num_gpus, rank=rank, shuffle=False, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, pin_memory=params.pin_memory, num_workers=opt.num_workers,
                                    drop_last=True, shuffle=False, sampler=val_sampler, collate_fn=BddDataset.collate_fn)
    
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    opt = get_args()
    opt.saved_path = opt.saved_path + f'/{opt.project}/'
    opt.log_path = opt.log_path + f'/{opt.project}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)
    print(1)
    mp.spawn(
        train,
        args=(opt,),
        nprocs=opt.num_gpus
    )