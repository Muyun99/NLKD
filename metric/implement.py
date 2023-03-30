import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(pred_segm, gt_segm):
    h_e, w_e = segm_size(pred_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def extract_both_masks(pred_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(pred_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(pred_segm, gt_segm):
    eval_cl, _ = extract_classes(pred_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)


class mIoU(nn.Module):
    def __init__(self):
        super(mIoU, self).__init__()

    def forward(self, pred_segm, gt_segm):
        '''
        (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
        '''
        check_size(pred_segm, gt_segm)
        cl, n_cl = union_classes(pred_segm, gt_segm)
        _, n_cl_gt = extract_classes(gt_segm)
        eval_mask, gt_mask = extract_both_masks(pred_segm, gt_segm, cl, n_cl)

        IU = list([0]) * n_cl

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]

            if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
                continue

            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            t_i = np.sum(curr_gt_mask)
            n_ij = np.sum(curr_eval_mask)

            IU[i] = n_ii / (t_i + n_ij - n_ii)

        mean_IU_ = np.sum(IU) / n_cl_gt
        return mean_IU_


class BFScore(nn.Module):
    def __init__(self):
        super(BFScore, self).__init__()

    def forward(self, pred, gt):
        """
               Input:
                   - pred: the output from model (before softmax)
                           shape (N, C, H, W)
                   - gt: ground truth map
                           shape (N, H, w)
               Return:
                   - boundary loss, averaged over mini-bathc
               """

        n, c, _, _ = pred.shape

        # boundary map
        gt_b = F.max_pool2d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        return BF1
