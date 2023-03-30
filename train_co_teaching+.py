import os
import sys
import torch
import argparse
import logging
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# dataset
from data.implement import BasicDataset_without_weight, train_transform
from torch.utils.data import DataLoader

# tensorboard & distrubuted
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# model
from model import UNet
from modeling.deeplab import *
from optimizer import optim_ranger
from scheduler import scheduler_linear
from loss import loss_bce
from utils import eval_net_unet_dice, eval_net_unet_bfscore, eval_net_unet_miou
from utils.weight_function import BoundaryScore_fast, jaccard_index



miou_func = jaccard_index()
bfscore_func = BoundaryScore_fast()

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []

    for pred, label in zip(torch.sigmoid(preds) > 0.5, torch.sigmoid(labels) > 0.5):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou, ious

# 利用 co_teaching+ 论文算法得到idx
def co_teaching_plus(mask_pred_teacher, mask_pred_student, true_masks, threshold=0.9, ratio=0.6):
    miou_teacher, list_miou_teacher = iou_binary(preds=mask_pred_teacher, labels=true_masks)
    miou_student, list_miou_student = iou_binary(preds=mask_pred_student, labels=true_masks)
    miou, list_miou = iou_binary(preds=mask_pred_teacher, labels=mask_pred_student)

    # find disagreement
    disagreement_idx = [i for i in range(len(list_miou)) if list_miou[i] < threshold]

    num_sample = int(len(list_miou_teacher) * ratio)
    # teacher 认为miou高的样本给 student
    idx_student = np.argsort(list_miou_teacher)[num_sample:]
    # student 认为miou高的样本给 teacher
    idx_teacher = np.argsort(list_miou_student)[num_sample:]

    final_idx_student = [idx for idx in idx_student if idx in disagreement_idx]
    final_idx_teacher = [idx for idx in idx_teacher if idx in disagreement_idx]

    return final_idx_student, final_idx_teacher


# 只对idx中的样本求loss后更新权重
class criterion_with_idx(nn.Module):
    def __init__(self):
        super(criterion_with_idx, self).__init__()

    def forward(self, true, pred, idx):
        loss = 0
        for idx_now, sample in enumerate(zip(true, pred)):
            sample_true = sample[0]
            sample_pred = sample[1]
            if idx_now in idx:
                sample_loss = loss_bce(sample_true, sample_pred)
                loss += sample_loss
        return loss / true.shape[0]


def train_decoupling(net_student,
                     net_teacher,
                     device,
                     epochs=5,
                     lr=0.1,
                     batch_size=8,
                     save_cp=True):
    global dir_checkpoint
    optimizer_teacher = optim_ranger(net_teacher.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler_teacher = scheduler_linear(optimizer_teacher, step_size=25, gamma=0.5)
    criterion_teacher = criterion_with_idx()

    optimizer_student = optim_ranger(net_student.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler_student = scheduler_linear(optimizer_student, step_size=25, gamma=0.5)
    criterion_student = criterion_with_idx()

    net_teacher.to(device)
    net_student.to(device)
    train_dataset = BasicDataset_without_weight(file_csv=args.train_csv,
                                                transform=train_transform)
    val_dataset = BasicDataset_without_weight(file_csv=args.valid_csv,
                                              transform=train_transform)
    test_dataset = BasicDataset_without_weight(file_csv=args.test_csv,
                                               transform=train_transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment="_{}".format(args.name))
    n_train = len(train_dataset)
    n_valid = len(val_dataset)

    logging.info(
        f'''Starting training:
                Epochs:          {epochs}
                Batch size:      {batch_size}
                Learning rate:   {lr}
                Training size:   {n_train}
                Validation size: {n_valid}
                Checkpoints:     {save_cp}
                Device:          {device}
                '''
    )

    global_step = 0
    val_score_student = 0
    best_valid_score_student = 0
    best_bfscore_score_student = 0
    best_miou_score_student = 0

    val_score_teacher = 0
    best_valid_score_teacher = 0
    best_bfscore_score_teacher = 0
    best_miou_score_teacher = 0

    idx_teacher = []
    idx_student = []
    epoch_smaple = 0
    for epoch in range(epochs):
        net_teacher.train()
        net_student.train()
        epoch_loss_student = 0
        epoch_loss_teacher = 0
        with tqdm(total=n_train,
                  desc='Epoch {}/{}/val_stu:{}/val_tea:{}/idx:{}'.format(epoch + 1, epochs, val_score_student,
                                                                         val_score_teacher, epoch_smaple,
                                                                         unit='img')) as pbar:
            for batch in train_dataloader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net_student.n_channels, \
                    'Network has been defined with {} input channels, '.format(
                        net_student.n_channels) + 'but loaded images have {} channels. Please check that '.format(
                        imgs.shape[1]) + 'the images are loaded correctly.'
                assert imgs.shape[1] == net_teacher.n_channels, \
                    'Network has been defined with {} input channels, '.format(
                        net_teacher.n_channels) + 'but loaded images have {} channels. Please check that '.format(
                        imgs.shape[1]) + 'the images are loaded correctly.'
                imgs = imgs.cuda(non_blocking=True)
                true_masks = true_masks.cuda(non_blocking=True)

                # pred and choose samples
                mask_pred_student = net_student(imgs)
                mask_pred_teacher = net_teacher(imgs)
                idx_student, idx_teacher = co_teaching_plus(mask_pred_teacher, mask_pred_student, true_masks)
                if len(idx_student) == 0:
                    idx_student = [0]
                if len(idx_teacher) == 0:
                    idx_teacher = [0]
                epoch_smaple += len(idx_student)

                # student training process
                loss_student = criterion_student(mask_pred_student, true_masks, idx_student)
                epoch_loss_student += loss_student.item()
                writer.add_scalar('Train/Loss_student', loss_student.item(), global_step=global_step)
                optimizer_student.zero_grad()
                loss_student.backward(retain_graph=True)
                nn.utils.clip_grad_value_(net_student.parameters(), 0.1)
                optimizer_student.step()

                # teacher training process
                loss_teacher = criterion_teacher(mask_pred_teacher, true_masks, idx_teacher)
                epoch_loss_teacher += loss_teacher.item()
                writer.add_scalar('Train/Loss_teacher', loss_teacher.item(), global_step=global_step)
                optimizer_teacher.zero_grad()
                loss_teacher.backward()
                nn.utils.clip_grad_value_(net_teacher.parameters(), 0.1)
                optimizer_teacher.step()

                pbar.set_postfix(
                    **{'loss_student (batch)': loss_student.item(), 'loss_teacher (batch)': loss_teacher.item()})
                pbar.update(imgs.shape[0])
                global_step += 1

            val_bfscore_student = eval_net_unet_bfscore(net_student, valid_dataloader, device)
            val_miouscore_student = eval_net_unet_miou(net_student, valid_dataloader, device)
            val_dicescore_student = eval_net_unet_dice(net_student, valid_dataloader, device)
            val_score_student = (val_bfscore_student + val_miouscore_student + val_dicescore_student) / 3

            val_bfscore_teacher = eval_net_unet_bfscore(net_teacher, valid_dataloader, device)
            val_miouscore_teacher = eval_net_unet_miou(net_teacher, valid_dataloader, device)
            val_dicescore_teacher = eval_net_unet_dice(net_teacher, valid_dataloader, device)
            val_score_teacher = (val_bfscore_teacher + val_miouscore_teacher + val_dicescore_teacher) / 3
            scheduler_student.step()
            scheduler_teacher.step()

            writer.add_scalar('Train/lr_student', optimizer_student.param_groups[0]['lr'], global_step=global_step)
            writer.add_scalar('Train/lr_teacher', optimizer_teacher.param_groups[0]['lr'], global_step=global_step)

            logging.info('Validation cross entropy for teacher: {}'.format(val_score_teacher))
            writer.add_scalar('Valid/val_score_teacher', val_score_teacher, global_step=global_step)
            logging.info('Validation cross entropy for student: {}'.format(val_score_student))
            writer.add_scalar('Valid/val_score_student', val_score_student, global_step=global_step)

        if save_cp:
            dir_checkpoint_now = os.path.join(dir_checkpoint, args.name)
            if not os.path.exists(dir_checkpoint_now):
                os.mkdir(dir_checkpoint_now)
                logging.info('Create checkopint directory')

            if val_score_teacher > best_valid_score_teacher:
                best_valid_score_teacher = val_score_teacher
                torch.save(net_teacher.state_dict(), os.path.join(dir_checkpoint_now, 'teacher_best.pth'))
                logging.info('Checkpoint {} saved!'.format(epoch + 1))
            if val_bfscore_teacher > best_bfscore_score_teacher:
                best_bfscore_score_teacher = val_bfscore_teacher
                torch.save(net_teacher.state_dict(), os.path.join(dir_checkpoint_now, 'teacher_bfscore_best.pth'))
                logging.info('bfscore best Checkpoint {} saved!'.format(epoch + 1))
            if val_miouscore_teacher > best_miou_score_teacher:
                best_miou_score_teacher = val_miouscore_teacher
                torch.save(net_teacher.state_dict(), os.path.join(dir_checkpoint_now, 'teacher_miou_best.pth'))
                logging.info('miou best Checkpoint {} saved!'.format(epoch + 1))

            if val_score_student > best_valid_score_student:
                best_valid_score_student = val_score_student
                torch.save(net_student.state_dict(), os.path.join(dir_checkpoint_now, 'student_best.pth'))
                logging.info('Checkpoint {} saved!'.format(epoch + 1))
            if val_bfscore_student > best_bfscore_score_student:
                best_bfscore_score_student = val_bfscore_student
                torch.save(net_student.state_dict(), os.path.join(dir_checkpoint_now, 'student_bfscore_best.pth'))
                logging.info('bfscore best Checkpoint {} saved!'.format(epoch + 1))
            if val_miouscore_student > best_miou_score_student:
                best_miou_score_student = val_miouscore_student
                torch.save(net_student.state_dict(), os.path.join(dir_checkpoint_now, 'student_miou_best.pth'))
                logging.info('miou best Checkpoint {} saved!'.format(epoch + 1))

    # get test_score from checkpoint with best valid_score
    net_teacher.load_state_dict(torch.load(os.path.join(dir_checkpoint_now, 'teacher_best.pth'), map_location=device))
    test_mIoU = eval_net_unet_miou(net_teacher, test_dataloader, device)
    logging.info('Teacher Test mIoU: {}'.format(test_mIoU))
    writer.add_scalar('test/teacher_mIoU', test_mIoU, global_step=global_step)

    test_dice = eval_net_unet_dice(net_teacher, test_dataloader, device)
    logging.info('Teacher Test Dice Coeff: {}'.format(test_dice))
    writer.add_scalar('test/teacher_Dice', test_dice, global_step=global_step)

    test_bfscore = eval_net_unet_bfscore(net_teacher, test_dataloader, device)
    logging.info('Teacher Test BFScore: {}'.format(test_bfscore))
    writer.add_scalar('test/teacher_BFScore', test_bfscore, global_step=global_step)

    net_teacher.load_state_dict(
        torch.load(os.path.join(dir_checkpoint_now, 'teacher_bfscore_best.pth'), map_location=device))
    test_best_bfscore = eval_net_unet_bfscore(net_teacher, test_dataloader, device)
    logging.info('Teacher Test Best BFScore: {}'.format(test_best_bfscore))
    writer.add_scalar('test/teacher_BFScore_best', test_best_bfscore, global_step=global_step)

    net_teacher.load_state_dict(
        torch.load(os.path.join(dir_checkpoint_now, 'teacher_miou_best.pth'), map_location=device))
    test_best_miou = eval_net_unet_miou(net_teacher, test_dataloader, device)
    logging.info('Teacher Test Best mIoU: {}'.format(test_best_miou))
    writer.add_scalar('test/teacher_mIoU_best', test_best_miou, global_step=global_step)

    net_student.load_state_dict(torch.load(os.path.join(dir_checkpoint_now, 'student_best.pth'), map_location=device))
    test_mIoU = eval_net_unet_miou(net_student, test_dataloader, device)
    logging.info('student Test mIoU: {}'.format(test_mIoU))
    writer.add_scalar('test/student_mIoU', test_mIoU, global_step=global_step)

    test_dice = eval_net_unet_dice(net_student, test_dataloader, device)
    logging.info('student Test Dice Coeff: {}'.format(test_dice))
    writer.add_scalar('test/student_Dice', test_dice, global_step=global_step)

    test_bfscore = eval_net_unet_bfscore(net_student, test_dataloader, device)
    logging.info('student Test BFScore: {}'.format(test_bfscore))
    writer.add_scalar('test/student_BFScore', test_bfscore, global_step=global_step)

    net_student.load_state_dict(
        torch.load(os.path.join(dir_checkpoint_now, 'student_bfscore_best.pth'), map_location=device))
    test_best_bfscore = eval_net_unet_bfscore(net_student, test_dataloader, device)
    logging.info('student Test Best BFScore: {}'.format(test_best_bfscore))
    writer.add_scalar('test/student_BFScore_best', test_best_bfscore, global_step=global_step)

    net_student.load_state_dict(
        torch.load(os.path.join(dir_checkpoint_now, 'student_miou_best.pth'), map_location=device))
    test_best_miou = eval_net_unet_miou(net_student, test_dataloader, device)
    logging.info('student Test Best mIoU: {}'.format(test_best_miou))
    writer.add_scalar('test/student_mIoU_best', test_best_miou, global_step=global_step)
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-train', '--train_csv', dest='train_csv', type=str, default=False,
                        help='train csv file_path')
    parser.add_argument('-valid', '--valid_csv', dest='valid_csv', type=str, default=False,
                        help='valid csv file_path')
    parser.add_argument('-test', '--test_csv', dest='test_csv', type=str, default=False,
                        help='test csv file_path')
    parser.add_argument('-n', '--name', dest='name', type=str, default="",
                        help='train name')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(filename=f'logs/{args.name}.log', level=logging.INFO, format='%(levelname)s: %(message)s')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpus = [0, 1]

    net_deeplab = DeepLab(num_classes=1, backbone='resnet', sync_bn=True)
    net_deeplab = torch.nn.DataParallel(net_deeplab.to(device), output_device=gpus[0])
    net_deeplab.n_classes = 1
    net_deeplab.n_channels = 3
    
    dir_checkpoint = 'checkpoints'

    # deeplab 加载50epoch的权重
    args.load = 'checkpoints/train_deeplabv3+_withoutweight_50/best.pth'
    net_deeplab.load_state_dict(torch.load(args.load, map_location=device))
    logging.info('Model loaded form {}'.format(args.load))

    net_unet = UNet(n_classes=1, n_channels=3)
    net_unet = torch.nn.DataParallel(net_unet.to(device), output_device=gpus[0])
    net_unet.n_classes = 1
    net_unet.n_channels = 3

    # unet 加载50epoch的权重
    args.load = 'checkpoints/train_iteration_6_50/best.pth'
    net_unet.load_state_dict(torch.load(args.load, map_location=device))
    logging.info('Model loaded form {}'.format(args.load))

    try:
        train_decoupling(net_student=net_unet,
                         net_teacher=net_deeplab,
                         device=device,
                         epochs=args.epochs,
                         batch_size=args.batchsize,
                         lr=args.lr)
    except KeyboardInterrupt:
        torch.save(net_unet.state_dict(), 'net_unet_INTERRUPTED.pth')
        torch.save(net_deeplab.state_dict(), 'net_deeplab_INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
