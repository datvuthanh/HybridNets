import torch
import numpy as np
import argparse
from tqdm.autonotebook import tqdm
import os

from utils import smp_metrics
from utils.utils import ConfusionMatrix, postprocess, scale_coords, process_batch, ap_per_class, fitness, \
    save_checkpoint, BBoxTransform, ClipBoxes, boolean_string, Params
from backbone import HybridNetsBackbone
from hybridnets.dataset import BddDataset
from torchvision import transforms
import torch.distributed as dist
import time


@torch.no_grad()
def val(model, rank, optimizer, val_generator, params, opt, writer, epoch, step, best_fitness, best_loss, best_epoch):
    model.eval()
    loss_regression_ls = []
    loss_classification_ls = []
    loss_segmentation_ls = []
    jdict, stats, ap, ap_class = [], [], [], []
    iou_thresholds = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    num_thresholds = iou_thresholds.numel()
    names = {i: v for i, v in enumerate(params.obj_list)}
    nc = len(names)
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    s = ('%15s' + '%11s' * 14) % (
    'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'mIoU', 'mF1', 'fIoU', 'sIoU', 'rIoU', 'rF1', 'lIoU', 'lF1')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    iou_ls = [[] for _ in range(3)]
    f1_ls = [[] for _ in range(3)]
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    progress_bar = tqdm(val_generator, ascii=True)
    for iter, data in enumerate(progress_bar):
        if rank == 0:
            progress_bar.update()
        imgs = data['img'].to(rank)
        annot = data['annot'].to(rank)
        seg_annot = data['segmentation'].to(rank)
        filenames = data['filenames']
        shapes = data['shapes']

        cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot,
                                                                                                seg_annot,
                                                                                                obj_list=params.obj_list)
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        seg_loss = seg_loss.mean()

        if opt.cal_map:
            out = postprocess(imgs.detach(),
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regression.detach(),
                              classification.detach(),
                              regressBoxes, clipBoxes,
                              0.001, 0.6)  # 0.5, 0.3

            for i in range(annot.size(0)):
                seen += 1
                labels = annot[i]
                labels = labels[labels[:, 4] != -1]

                ou = out[i]
                nl = len(labels)

                pred = np.column_stack([ou['rois'], ou['scores']])
                pred = np.column_stack([pred, ou['class_ids']])
                pred = torch.from_numpy(pred).cuda()

                target_class = labels[:, 4].tolist() if nl else []  # target class

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, num_thresholds, dtype=torch.bool),
                                      torch.Tensor(), torch.Tensor(), target_class))
                    # print("here")
                    continue

                if nl:
                    pred[:, :4] = scale_coords(imgs[i][1:], pred[:, :4], shapes[i][0], shapes[i][1])
                    labels = scale_coords(imgs[i][1:], labels, shapes[i][0], shapes[i][1])
                    correct = process_batch(pred, labels, iou_thresholds)
                    if opt.plots:
                        confusion_matrix.process_batch(pred, labels)
                else:
                    correct = torch.zeros(pred.shape[0], num_thresholds, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))

                # print(stats)

                # Visualization
                # seg_0 = segmentation[i]
                # # print('bbb', seg_0.shape)
                # seg_0 = torch.argmax(seg_0, dim = 0)
                # # print('before', seg_0.shape)
                # seg_0 = seg_0.cpu().numpy()
                #     #.transpose(1, 2, 0)
                # # print(seg_0.shape)
                # anh = np.zeros((384,640,3))
                # anh[seg_0 == 0] = (255,0,0)
                # anh[seg_0 == 1] = (0,255,0)
                # anh[seg_0 == 2] = (0,0,255)
                # anh = np.uint8(anh)
                # cv2.imwrite('segmentation-{}.jpg'.format(filenames[i]),anh)

            # Convert segmentation tensor --> 3 binary 0 1
            # batch_size, num_classes, height, width
            _, segmentation = torch.max(segmentation, 1)
            # _, seg_annot = torch.max(seg_annot, 1)
            seg = torch.zeros((seg_annot.size(0), 3, 384, 640), dtype=torch.int32)
            seg[:, 0, ...][segmentation == 0] = 1
            seg[:, 1, ...][segmentation == 1] = 1
            seg[:, 2, ...][segmentation == 2] = 1

            tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(seg.cuda(), seg_annot.long().cuda(),
                                                                   mode='multilabel', threshold=None)

            iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
            #         print(iou)
            f1 = smp_metrics.balanced_accuracy(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')

            for i in range(len(params.seg_list) + 1):
                iou_ls[i].append(iou.T[i].detach().cpu().numpy())
                f1_ls[i].append(f1.T[i].detach().cpu().numpy())

        loss = cls_loss + reg_loss + seg_loss
        if loss == 0 or not torch.isfinite(loss):
            continue

        loss_classification_ls.append(cls_loss.item())
        loss_regression_ls.append(reg_loss.item())
        loss_segmentation_ls.append(seg_loss.item())

    cls_loss = np.mean(loss_classification_ls)
    reg_loss = np.mean(loss_regression_ls)
    seg_loss = np.mean(loss_segmentation_ls)
    loss = cls_loss + reg_loss + seg_loss

    if rank == 0:
        print(
            'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Segmentation loss: {:1.5f}. Total loss: {:1.5f}'.format(
                epoch, opt.num_epochs, cls_loss, reg_loss, seg_loss, loss))
        writer.add_scalars('Loss', {'val': loss}, step)
        writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
        writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
        writer.add_scalars('Segmentation_loss', {'val': seg_loss}, step)

        ddp_stats = [None for _ in range(opt.num_gpus)]
        ddp_iou = [None for _ in range(opt.num_gpus)]
        ddp_f1 = [None for _ in range(opt.num_gpus)]
        # ddp_iou_first_decoder = [None for _ in range(opt.num_gpus)]
        # ddp_iou_second_decoder = [None for _ in range(opt.num_gpus)]    

        dist.gather_object(stats, ddp_stats, dst=0)
        dist.gather_object(iou_ls, ddp_iou, dst=0)
        dist.gather_object(f1_ls, ddp_f1, dst=0)
        # dist.gather_object(ddp_iou_first_decoder, iou_ls[0] + iou_ls[1], dst=0)
        # dist.gather_object(ddp_iou_second_decoder, iou_ls[0] + iou_ls[2], dst=0)
    else:
        dist.gather_object(stats, dst=0)
        dist.gather_object(iou_ls, dst=0)
        dist.gather_object(f1_ls, dst=0)


    if opt.cal_map and rank == 0:
        start_time = time.time()
        stats = [x for ranking in ddp_stats for x in ranking]
        iou_ls = [[] for _ in range(3)]
        for rank in ddp_iou:
            for rank_iou in rank:
                for i in range(3):
                    iou_ls[i].extend(rank_iou[i])
        f1_ls = [[] for _ in range(3)]
        for rank in ddp_f1:
            for rank_f1 in rank:
                for i in range(3):
                    f1_ls[i].extend(rank_f1[i])
        # print("LOOP: %s seconds" % (time.time() - start_time))

        # print(len(iou_ls[0]))
        iou_score = np.mean(iou_ls)
        # print(iou_score)
        f1_score = np.mean(f1_ls)

        iou_first_decoder = iou_ls[0] + iou_ls[1]
        iou_first_decoder = np.mean(iou_first_decoder)

        iou_second_decoder = iou_ls[0] + iou_ls[2]
        iou_second_decoder = np.mean(iou_second_decoder)

        for i in range(len(params.seg_list) + 1):
            iou_ls[i] = np.mean(iou_ls[i])
            f1_ls[i] = np.mean(f1_ls[i])

        # Compute statistics

        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        # print("UNZIP STATS: %s seconds" % (time.time() - start_time))

        # print(stats[3])

        # Count detected boxes per class
        # boxes_per_class = np.bincount(stats[2].astype(np.int64), minlength=1)

        ap50 = None
        save_dir = 'plots'
        os.makedirs(save_dir, exist_ok=True)

        # Compute metrics
        if len(stats) and stats[0].any():
            p, r, f1, ap, ap_class = ap_per_class(*stats, plot=opt.plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=1)  # number of targets per class
        else:
            nt = torch.zeros(1)
        # print("CAL MAP: %s seconds" % (time.time() - start_time))

        # Print results
        print(s)
        pf = '%15s' + '%11i' * 2 + '%11.3g' * 12  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map, iou_score, f1_score, iou_first_decoder, iou_second_decoder,
                    iou_ls[1], f1_ls[1], iou_ls[2], f1_ls[2]))

        # Print results per class
        training = True
        if (opt.verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
            pf = '%15s' + '%11i' * 2 + '%11.3g' * 4
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Plots
        if opt.plots:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
            confusion_matrix.tp_fp()

        results = (mp, mr, map50, map, iou_score, f1_score, loss)
        fi = fitness(
            np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95, iou, f1, loss ]

        # if calculating map, save by best fitness
        if fi > best_fitness:
            best_fitness = fi
            ckpt = {'epoch': epoch,
                    'step': step,
                    'best_fitness': best_fitness,
                    'model': model,
                    # 'optimizer': optimizer.state_dict()
                    }
            print("Saving checkpoint with best fitness", fi[0])
            save_checkpoint(ckpt, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}_best.pth')
    else:
        pass
        # if not calculating map, save by best loss
        # if loss + opt.es_min_delta < best_loss:
        #     best_loss = loss
        #     best_epoch = epoch

        #     save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}_best.pth')

    # Early stopping
    if epoch - best_epoch > opt.es_patience > 0:
        print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
        writer.close()
        exit(0)

    model.train()
    return best_fitness, best_loss, best_epoch