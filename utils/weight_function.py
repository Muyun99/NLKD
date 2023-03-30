import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import mmcv
import torch.nn.functional as F
from scipy.spatial import distance
from tqdm import tqdm
from config.noisy_dataset_path import *
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

bDebug = False
major = cv2.__version__.split('.')[0]  # Get opencv version


class single_class_bfscore_gpu(nn.Module):
    def __init__(self, threshold=2):
        super(single_class_bfscore_gpu, self).__init__()
        self.threshold = threshold

    """ For precision, contours_a==GT & contours_b==Prediction
        For recall, contours_a==Prediction & contours_b==GT """

    # precision, contours_a==GT & contours_b==Prediction
    # recall, contours_a==Prediction & contours_b==GT
    def calc_precision_recall(self, contours_a, contours_b, threshold):
        top_count = 0
        length_b = len(contours_b)

        try:
            dis = distance.cdist(contours_b, contours_a, 'euclidean')

            dis[dis < threshold] = 1
            dis[dis >= threshold] = 0

            for b in range(length_b):
                single_dis_list = dis[b]
                if sum(single_dis_list) > 0:
                    top_count = top_count + 1

            precision_recall = top_count / length_b
            # debug= 1
        except Exception as e:
            precision_recall = 0

        return precision_recall, top_count, length_b

    """ computes the BF (Boundary F1) contour matching score between the predicted and GT segmentation """

    def forward(self, gtfile, prfile):
        # 转为语义分割的标注格式
        gt_ = mmcv.imread(gtfile, flag='grayscale')
        pr_ = mmcv.imread(prfile, flag='grayscale')

        _, gt_ = cv2.threshold(gt_, 1, 255, cv2.THRESH_BINARY)
        _, pr_ = cv2.threshold(pr_, 1, 255, cv2.THRESH_BINARY)

        # 检查 GT 和 prediction 的类别是否相同
        classes_gt = np.unique(gt_)  # Get GT classes
        classes_pr = np.unique(pr_)  # Get predicted classes

        if not np.array_equiv(classes_gt, classes_pr):
            classes = np.concatenate((classes_gt, classes_pr))
            classes = np.unique(classes)
            classes = np.sort(classes)
        else:
            classes = classes_gt  # Get matched classes

        # 每个类一个bfscore
        # Get the number of classes
        m = np.max(classes)
        # Define bfscore variable (initialized with zeros)
        bfscores = np.zeros((m + 1), dtype=float)
        areas_gt = np.zeros((m + 1), dtype=float)

        for i in range(m + 1):
            bfscores[i] = np.nan
            areas_gt[i] = np.nan

        for target_class in classes:  # Iterate over classes

            if target_class == 0:  # Skip background
                continue

            # 计算 GT 的边界
            gt = gt_.copy()
            gt[gt != target_class] = 0

            # contours 是 boundary point 的 list.
            if major == '3':  # For opencv version 3.x
                _, contours_Umat, _ = cv2.findContours(
                    cv2.UMat(gt), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Find contours of the shape
            else:  # For other opencv versions
                contours_Umat, _ = cv2.findContours(
                    cv2.UMat(gt), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Find contours of the shape

            # contours 是 list of numpy arrays
            # contours = cv2.UMat.get(contours_Umat)
            # contours_Umat.copyTo(contours)
            # print(type(contours_Umat[0]))
            contours_gt = []
            for i in range(len(contours_Umat)):
                for j in range(len(contours_Umat[i])):
                    contours_gt.append(contours_Umat[i][j][0].tolist())

            # Get contour area of GT
            if contours_gt:
                area = cv2.contourArea(np.array(contours_gt))
                areas_gt[target_class] = area

            # 计算 Prediction 的边界
            prediction = pr_.copy()
            prediction[prediction != target_class] = 0

            if major == '3':  # For opencv version 3.x
                _, contours, _ = cv2.findContours(
                    cv2.UMat(prediction), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            else:  # For other opencv versions
                contours, _ = cv2.findContours(
                    cv2.UMat(prediction), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            contours_pr = []
            for i in range(len(contours)):
                for j in range(len(contours[i])):
                    contours_pr.append(contours[i][j][0].tolist())

            precision, numerator, denominator = self.calc_precision_recall(
                contours_gt, contours_pr, self.threshold)  # Precision

            recall, numerator, denominator = self.calc_precision_recall(
                contours_pr, contours_gt, self.threshold)  # Recall

            f1 = 0
            try:
                f1 = 2 * recall * precision / (recall + precision)  # F1 score
            except:
                f1 = 0
            bfscores[target_class] = f1
        score = np.nanmean(bfscores[1:])
        if np.isnan(score):
            return 0
        else:
            return score


class diceCoeff(nn.Module):
    """ computational formula：
            dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    def __init__(self, eps, activation):
        super(diceCoeff, self).__init__()
        self.eps = eps
        self.activation = activation

    def forward(self, gtfile, prfile):
        gt = cv2.imread(gtfile)  # Read GT segmentation
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)  # Convert color space

        pred = cv2.imread(prfile)  # Read predicted segmentation
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)  # Convert color space

        gt = torch.Tensor(gt)
        pred = torch.Tensor(pred)

        if self.activation is None or self.activation == "none":
            activation_fn = lambda x: x
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
        elif self.activation == "softmax2d":
            activation_fn = nn.Softmax2d()
        else:
            raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

        pred = activation_fn(pred)

        N = gt.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)

        tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
        fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
        fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)
        # 转为float，以防long类型之间相除结果为0
        loss = (2 * tp + self.eps).float() / (2 * tp + fp + fn + self.eps).float()

        return (loss.sum() / N).numpy()


class jaccard_index(nn.Module):
    def __init__(self):
        super(jaccard_index, self).__init__()

    def intersect_and_union(self, pred_label, label, num_classes):
        """Calculate intersection and Union.
        Args:
            pred_label (ndarray): Prediction segmentation map
            label (ndarray): Ground truth segmentation map
            num_classes (int): Number of categories
            ignore_index (int): Index that will be ignored in evaluation.
         Returns:
             ndarray: The intersection of prediction and ground truth histogram
                 on all classes
             ndarray: The union of prediction and ground truth histogram on all
                 classes
             ndarray: The prediction histogram on all classes.
             ndarray: The ground truth histogram on all classes.
        """

        intersect = pred_label[pred_label == label]
        area_intersect, _ = np.histogram(
            intersect, bins=np.arange(num_classes + 1))
        area_pred_label, _ = np.histogram(
            pred_label, bins=np.arange(num_classes + 1))
        area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
        area_union = area_pred_label + area_label - area_intersect

        return area_intersect, area_union, area_pred_label, area_label

    def mean_iou(self, gt, pred, num_classes):
        """Calculate Intersection and Union (IoU)
        Args:
            results (list[ndarray]): List of prediction segmentation maps
            gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
            num_classes (int): Number of categories
            ignore_index (int): Index that will be ignored in evaluation.
            nan_to_num (int, optional): If specified, NaN values will be replaced
                by the numbers defined by the user. Default: None.
         Returns:
             float: Overall accuracy on all images.
             ndarray: Per category accuracy, shape (num_classes, )
             ndarray: Per category IoU, shape (num_classes, )
        """

        total_area_intersect = np.zeros((num_classes,), dtype=np.float)
        total_area_union = np.zeros((num_classes,), dtype=np.float)
        total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
        total_area_label = np.zeros((num_classes,), dtype=np.float)
        area_intersect, area_union, area_pred_label, area_label = \
            self.intersect_and_union(pred, gt, num_classes)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
        all_acc = total_area_intersect.sum() / total_area_label.sum()
        acc = total_area_intersect / total_area_label
        iou = total_area_intersect / total_area_union
        return all_acc, acc, iou

    def forward(self, mask_gt, mask_pred):
        all_acc, acc, miou = self.mean_iou(gt=mask_gt, pred=mask_pred, num_classes=1)
        return miou[0]


class boundary_jiscore(nn.Module):
    def __init__(self, threshold=4):
        self.threshold = threshold
        super(boundary_jiscore, self).__init__()

    def get_counter(self, img):
        if major == '3':  # For opencv version 3.x
            _, contours, _ = cv2.findContours(
                cv2.UMat(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:  # For other opencv versions
            contours, _ = cv2.findContours(
                cv2.UMat(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contours_list = []
        for i in range(len(contours)):
            for j in range(len(cv2.UMat.get(contours[i]))):
                contours_list.append(cv2.UMat.get(contours[i])[j][0].tolist())
        return contours_list

    def get_TP(self, contours, point):
        matrix_dis = distance.cdist(contours, point, 'euclidean')
        list_min_dis = np.min(matrix_dis, axis=1)
        list_min_dis[list_min_dis >= self.threshold] = self.threshold

        list_ones = np.ones(list_min_dis.shape)
        list_min_dis = (list_min_dis / self.threshold) * (list_min_dis / self.threshold)

        return (list_ones - list_min_dis).sum()

    def forward(self, gtfile, prfile):
        # gt = cv2.imread(gtfile)  # Read GT segmentation
        # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)  # Convert color space
        #
        # pred = cv2.imread(prfile)  # Read predicted segmentation
        # pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)  # Convert color space

        gt = mmcv.imread(gtfile, flag='grayscale')
        pred = mmcv.imread(prfile, flag='grayscale')

        point_gt = np.argwhere(gt == 255)
        point_pred = np.argwhere(pred == 255)
        contours_gt = self.get_counter(gt)
        contours_pred = self.get_counter(pred)
        if len(contours_gt) == 0 or len(point_gt) == 0 or len(contours_pred) == 0 or len(point_pred) == 0:
            return 0
        else:
            TP_gt = self.get_TP(contours_gt, point_pred)
            TP_pred = self.get_TP(contours_pred, point_gt)
            FN = len(contours_gt) - TP_gt
            FP = len(contours_pred) - TP_pred

            BJ = (TP_gt + TP_pred) / (TP_gt + TP_pred + FN + FP)
            return BJ


# BFScore的fast版本
class BoundaryScore_fast(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def one_hot(self, label, n_classes, requires_grad=True):
        """Return One Hot Label"""
        one_hot_label = torch.eye(
            n_classes, device=self.device, requires_grad=requires_grad)[label]
        one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

        return one_hot_label

    def forward(self, gt, pred):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _  = pred.shape

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


if __name__ == '__main__':
    bfscore_fast = BoundaryScore_fast()
    bfscore_slow = single_class_bfscore_gpu()

    bfscore_value_fast_list = []
    bfscore_value_slow_list = []
    for i in tqdm(range(1, 10)):
        gtfile = os.path.join(mask_path_all, f'{i}.png')
        prfile = os.path.join(mask_pointrend_path, f'{i}.png')

        gt = mmcv.imread(gtfile, flag='grayscale')
        pred = mmcv.imread(prfile, flag='grayscale')

        gt = gt.reshape((1, 1, 321, 321))
        pred = pred.reshape((1, 1, 321, 321))

        gt = torch.from_numpy(gt / 255).float()
        pred = torch.from_numpy(pred / 255).float()

        bfscore_value_fast = bfscore_fast(pred, gt)
        bfscore_value_slow = bfscore_slow(gtfile, prfile)

        bfscore_value_fast_list.append(bfscore_value_fast * 100)
        bfscore_value_slow_list.append(bfscore_value_slow * 100)

    m = {}
    m['fast'] = 'o'
    m['slow'] = '>'
    ratios = range(1, 10)

    fig, ax = plt.subplots(dpi=500)
    noise_Normal_plot = plt.errorbar(ratios, bfscore_value_fast_list, label='fast', marker=m['fast'])
    noise_bfscore_th2_plot = plt.errorbar(ratios, bfscore_value_slow_list, label='slow', marker=m['slow'])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
    plt.legend(handles=[noise_Normal_plot, noise_bfscore_th2_plot], loc=4)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)

    plt.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    plt.xlabel('% of Noisy Data')
    plt.ylabel('Mean IoU (%)')
    plt.title('Supervisely')
    plt.grid(True)
    plt.show()

    # print(f'{i}: {bfscore_value_fast}, {bfscore_value_slow}')
