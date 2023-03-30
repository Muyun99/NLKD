import os
import sys
import torch
import argparse
import logging
import torch.nn as nn
from tqdm import tqdm

# dataset
from data.implement import BasicDataset_with_weight, train_transform, BasicDataset_without_weight
from torch.utils.data import DataLoader

# tensorboard & distrubuted
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# model
from model import UNet
from optimizer import optim_ranger
from scheduler import scheduler_linear
from loss import loss_bce
from utils import eval_net_unet_dice, eval_net_unet_miou, eval_net_unet_bfscore
from torch.cuda.amp import GradScaler, autocast




class criterion(nn.Module):
    # zero block 是将权重为 0 的样本赋值为一个固定值
    def __init__(self, zero_value):
        super(criterion, self).__init__()
        self.zero_value = zero_value

    def forward(self, true, pred, score):
        loss = 0
        for idx, sample in enumerate(zip(true, pred, score)):
            sample_true = sample[0]
            sample_pred = sample[1]
            sample_score = sample[2]
            sample_loss = loss_func(sample_true, sample_pred)
            if sample_score == 0 or torch.isnan(sample_score):
                sample_score = self.zero_value
            loss += sample_score * sample_loss

        return loss / true.shape[0]


def train_net(net,
              device,
              epochs=5,
              lr=0.1,
              batch_size=8,
              save_cp=True):
    global dir_checkpoint
    net.to(device)
    train_dataset = BasicDataset_with_weight(file_csv=args.train_csv,
                                             transform=train_transform)
    val_dataset = BasicDataset_with_weight(file_csv=args.valid_csv,
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

    global_step = 0
    best_valid_score = 0
    best_miou_score = 0
    best_bfscore_score = 0
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

    scaler = GradScaler()
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch + 1, epochs), unit='img') as pbar:
            for batch in train_dataloader:
                imgs = batch['image']
                true_masks = batch['mask']
                weight = batch['weight']
                assert imgs.shape[1] == net.n_channels, \
                    'Network has been defined with {} input channels, '.format(
                        net.n_channels) + 'but loaded images have {} channels. Please check that '.format(
                        imgs.shape[1]) + 'the images are loaded correctly.'

                imgs = imgs.cuda(non_blocking=True)
                true_masks = true_masks.cuda(non_blocking=True)
                weight = weight.cuda(non_blocking=True)

                optimizer.zero_grad()
                fp16 = False
                if fp16 is True:
                    with autocast():
                        mask_pred = net(imgs)
                        loss = criterion_func(mask_pred, true_masks, weight)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    mask_pred = net(imgs)
                    loss = criterion_func(mask_pred, true_masks, weight)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                pbar.update(imgs.shape[0])
                global_step += 1

            val_bfscore = eval_net_unet_bfscore(net, valid_dataloader, device)
            val_miouscore =  eval_net_unet_miou(net, valid_dataloader, device)
            val_dicescore = eval_net_unet_dice(net, valid_dataloader, device) 
            val_score = (val_bfscore + val_miouscore + val_dicescore) / 3

            scheduler.step()
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step=global_step)

            logging.info('Validation score: {}'.format(val_score))
            writer.add_scalar('Valid/val_score', val_score, global_step=global_step)

        if save_cp:
            dir_checkpoint_now = os.path.join(dir_checkpoint, args.name)
            if not os.path.exists(dir_checkpoint_now):
                os.mkdir(dir_checkpoint_now)
                logging.info('Create checkopint directory')
            if val_score > best_valid_score:
                best_valid_score = val_score
                torch.save(net.state_dict(), os.path.join(dir_checkpoint_now, 'best.pth'))
                logging.info('Checkpoint {} saved!'.format(epoch + 1))
            if val_bfscore > best_bfscore_score:
                best_bfscore_score = val_bfscore
                torch.save(net.state_dict(), os.path.join(dir_checkpoint_now, 'bfscore_best.pth'))
                logging.info('bfscore best Checkpoint {} saved!'.format(epoch + 1))
            if val_miouscore > best_miou_score:
                best_miou_score = val_miouscore
                torch.save(net.state_dict(), os.path.join(dir_checkpoint_now, 'miou_best.pth'))
                logging.info('miou best Checkpoint {} saved!'.format(epoch + 1))


        writer.add_scalar('Train/Loss', epoch_loss / n_train, global_step=global_step)

    net.load_state_dict(torch.load(os.path.join(dir_checkpoint_now, 'best.pth'), map_location=device))
    test_mIoU = eval_net_unet_miou(net, test_dataloader, device)
    logging.info('Test mIoU: {}'.format(test_mIoU))
    writer.add_scalar('test/mIoU', test_mIoU, global_step=global_step)

    test_dice = eval_net_unet_dice(net, test_dataloader, device)
    logging.info('Test Dice Coeff: {}'.format(test_dice))
    writer.add_scalar('test/Dice', test_dice, global_step=global_step)

    test_bfscore = eval_net_unet_bfscore(net, test_dataloader, device)
    logging.info('Test BFScore: {}'.format(test_bfscore))
    writer.add_scalar('test/BFScore', test_bfscore, global_step=global_step)

    net.load_state_dict(torch.load(os.path.join(dir_checkpoint_now, 'bfscore_best.pth'), map_location=device))
    test_best_bfscore = eval_net_unet_bfscore(net, test_dataloader, device)
    logging.info('Test BFScore: {}'.format(test_best_bfscore))
    writer.add_scalar('test/BFScore_best', test_best_bfscore, global_step=global_step)

    net.load_state_dict(torch.load(os.path.join(dir_checkpoint_now, 'miou_best.pth'), map_location=device))
    test_best_miou = eval_net_unet_miou(net, test_dataloader, device)
    logging.info('Test mIoU: {}'.format(test_best_miou))
    writer.add_scalar('test/mIoU_best', test_best_miou, global_step=global_step)

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

    net = UNet(n_classes=1, n_channels=3)
    net.n_classes = 1
    net.n_channels = 3

    dir_checkpoint = 'checkpoints'

    loss_dict = {}

    optimizer = optim_ranger(net.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = scheduler_linear(optimizer, step_size=25, gamma=0.5)
    loss_func = loss_bce
    criterion_func = criterion(zero_value=0)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info('Model loaded form {}'.format(args.load))

    try:
        train_net(net=net,
                  device=device,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
