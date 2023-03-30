import os
import math
import cv2
import torch
import time
import numpy as np
import pandas as pd
import mmcv
from tqdm import tqdm
from config.noisy_dataset_path import *
from utils.weight_function import single_class_bfscore_gpu, diceCoeff, jaccard_index, boundary_jiscore, BoundaryScore_fast


def get_csv(csv_train, csv_weight_save, weight_func, mask_teachernet_path):
    df_train = pd.read_csv(csv_train)
    dict_score = {}
    start_time = time.time()
    for idx in tqdm(range(len(df_train))):
        img_file = df_train.iloc[idx, 0]
        sample_gt = df_train.iloc[idx, 1]

        # for supervisely dataset
        sample_gt_name = sample_gt.split('/')[-1]
        sample_pred = os.path.join(mask_teachernet_path, sample_gt_name)

        # for cityscapes dataset
        # sample_gt_name_list = sample_gt.split('/')[-1].split('_')
        # sample_gt_name = f'{sample_gt_name_list[0]}_{sample_gt_name_list[1]}_{sample_gt_name_list[2]}_leftImg8bit.png'
        # sample_pred = os.path.join(mask_teachernet_path, sample_gt_name)

        mask_gt = mmcv.imread(sample_gt, channel_order='rgb')
        mask_pred = mmcv.imread(sample_pred, channel_order='rgb')

        if isinstance(weight_func, BoundaryScore_fast):
            gt = mask_gt.reshape((1, 1, 321, 321))
            pred = mask_pred.reshape((1, 1, 321, 321))
            gt = torch.from_numpy(gt / 255).float()
            pred = torch.from_numpy(pred / 255).float()
            score = weight_func(gtfile=gt, prfile=pred)
            dict_score[img_file] = [sample_gt, score.numpy()[0][0]]
        else:
            score = weight_func(mask_gt, mask_pred)
            dict_score[img_file] = [sample_gt, score]



    end_time = time.time()
    print(f'time is {end_time - start_time}')
    df_score = pd.DataFrame.from_dict(dict_score, orient='index')
    df_score.to_csv(csv_weight_save)


def mix_weight(csv_weight1, csv_weight2, csv_weight_save):
    pd_weight1 = pd.read_csv(csv_weight1)
    pd_weight2 = pd.read_csv(csv_weight2)
    dict_mix_weight = {}
    for idx in range(len(pd_weight1)):
        mix_weight_value = (pd_weight1.iloc[idx, 2] + pd_weight2.iloc[idx, 2]) / 2
        dict_mix_weight[pd_weight1.iloc[idx, 0]] = [pd_weight1.iloc[idx, 1], mix_weight_value]

    df_mix_weight = pd.DataFrame.from_dict(dict_mix_weight, orient='index')
    df_mix_weight.to_csv(csv_weight_save)
    print('mix done!')


if __name__ == "__main__":
    bfscore = BoundaryScore_fast()
    miou = jaccard_index()
    # dice = diceCoeff(eps=1e-5, activation='sigmoid')
    # bjscore = boundary_jiscore(threshold=4)

    # Supervisely Clean 的weight

    # get_csv(csv_train_clean, csv_train_clean_bfweight,
    #         weight_func=bfscore, mask_teachernet_path=mask_pointrend_path)

    # get_csv(csv_train_clean, csv_train_clean_miouweight,
    #         weight_func=miou, mask_teachernet_path=mask_pointrend_path)
    #
    # mix_weight(csv_train_clean_bfweight,
    #            csv_train_clean_miouweight,
    #            csv_train_clean_bfmiouweight)

    # # Supervisely Clean 的 weight
    # get_csv(csv_cityscapes_clean_train, csv_train_clean_bfweight,
    #         weight_func=bfscore, mask_teachernet_path=mask_deeplabv3_path)
    # get_csv(csv_train_clean, csv_train_clean_miouweight,
    #         weight_func=miou, mask_teachernet_path=mask_deeplabv3_path)
    # mix_weight(csv_train_clean_bfweight, csv_train_clean_miouweight,
    #            csv_train_clean_bfmiouweight)

    # Supervisely Deeplab3p Teacher
    # get_csv(train_csv_iteration_6_noisy_5000_50_train, train_csv_iteration_6_noisy_5000_50_train_Deeplab3p_bfweight,
    #         weight_func=bfscore, mask_teachernet_path=mask_deeplabv3_path)
    # get_csv(train_csv_iteration_6_noisy_5000_50_train, train_csv_iteration_6_noisy_5000_50_train_Deeplab3p_miouweight,
    #         weight_func=miou, mask_teachernet_path=mask_deeplabv3_path)
    # mix_weight(train_csv_iteration_6_noisy_5000_50_train_Deeplab3p_bfweight,
    #            train_csv_iteration_6_noisy_5000_50_train_Deeplab3p_miouweight,
    #            train_csv_iteration_6_noisy_5000_50_train_Deeplab3p_miou_bf_weight)

    # mix_weight(train_csv_iteration_6_noisy_5000_50_train_HRNet_miou_bf_weight,
    #            train_csv_iteration_6_noisy_5000_50_train_Deeplab3p_miou_bf_weight,
    #            train_csv_iteration_6_noisy_5000_50_train_HRNet_Deeplabv3p_miou_bf_weight)
    #
    # mix_weight(train_csv_iteration_6_noisy_5000_50_train_ocrnet_miou_bf_weight,
    #            train_csv_iteration_6_noisy_5000_50_train_miou_bf_weight,
    #            train_csv_iteration_6_noisy_5000_50_train_ocrnet_pointrend_miou_bf_weight)

    mix_weight(train_csv_iteration_6_noisy_5000_50_train_ocrnet_pointrend_miou_bf_weight,
               train_csv_iteration_6_noisy_5000_50_train_HRNet_Deeplabv3p_miou_bf_weight,
               train_csv_iteration_6_noisy_5000_50_train_HRNet_Deeplabv3p_ocrnet_pointrend_miou_bf_weight)

    # Supervisely HRNet Teacher
    # get_csv(train_csv_iteration_6_noisy_5000_50_train, train_csv_iteration_6_noisy_5000_50_train_HRNet_bfweight,
    #         weight_func=bfscore, mask_teachernet_path=mask_deeplabv3_path)
    # get_csv(train_csv_iteration_6_noisy_5000_50_train, train_csv_iteration_6_noisy_5000_50_train_HRNet_miouweight,
    #         weight_func=miou, mask_teachernet_path=mask_deeplabv3_path)
    # mix_weight(train_csv_iteration_6_noisy_5000_50_train_HRNet_bfweight,
    #            train_csv_iteration_6_noisy_5000_50_train_HRNet_miouweight,
    #            train_csv_iteration_6_noisy_5000_50_train_HRNet_miou_bf_weight)

    # Cityscapes clean的weight
    # get_csv(csv_cityscapes_full_coarse_train, csv_cityscapes_full_coarse_train_bfweight,
    #         weight_func=bfscore, mask_teachernet_path=cityscapes_pointrend_path)
    #
    # get_csv(csv_cityscapes_full_coarse_train, csv_cityscapes_full_coarse_train_miouweight,
    #         weight_func=miou, mask_teachernet_path=cityscapes_pointrend_path)
    #
    # mix_weight(csv_cityscapes_full_coarse_train_bfweight,
    #            csv_cityscapes_full_coarse_train_miouweight,
    #            csv_cityscapes_full_coarse_train_bfmiouweight)

    # Cityscapes coarse的weight
    # get_csv(csv_cityscapes_clean_train, csv_cityscapes_clean_train_bfweight, weight_func=bfscore,
    #         mask_teachernet_path=cityscapes_pointrend_path)
    # get_csv(csv_cityscapes_clean_train, csv_cityscapes_clean_train_miouweight, weight_func=miou,
    #         mask_teachernet_path=cityscapes_pointrend_path)
    # mix_weight(csv_cityscapes_clean_train_bfweight, csv_cityscapes_clean_train_miouweight,
    #            csv_cityscapes_clean_train_bfmiouweight)

    #

    # OCRNet 的 weight
    # get_csv(train_csv_iteration_6_noisy_5000_10_train, train_csv_iteration_6_noisy_5000_10_train_ocrnet_bfweight,
    #         weight_func=bfscore)
    # get_csv(train_csv_iteration_6_noisy_5000_20_train, train_csv_iteration_6_noisy_5000_20_train_ocrnet_bfweight,
    #         weight_func=bfscore)
    # get_csv(train_csv_iteration_6_noisy_5000_30_train, train_csv_iteration_6_noisy_5000_30_train_ocrnet_bfweight,
    #         weight_func=bfscore)
    # get_csv(train_csv_iteration_6_noisy_5000_40_train, train_csv_iteration_6_noisy_5000_40_train_ocrnet_bfweight,
    #         weight_func=bfscore)
    # get_csv(train_csv_iteration_6_noisy_5000_50_train, train_csv_iteration_6_noisy_5000_50_train_ocrnet_bfweight,
    #         weight_func=bfscore)
    #
    # get_csv(train_csv_iteration_6_noisy_5000_10_train, train_csv_iteration_6_noisy_5000_10_train_ocrnet_miouweight,
    #         weight_func=miou)
    # get_csv(train_csv_iteration_6_noisy_5000_20_train, train_csv_iteration_6_noisy_5000_20_train_ocrnet_miouweight,
    #         weight_func=miou)
    # get_csv(train_csv_iteration_6_noisy_5000_30_train, train_csv_iteration_6_noisy_5000_30_train_ocrnet_miouweight,
    #         weight_func=miou)
    # get_csv(train_csv_iteration_6_noisy_5000_40_train, train_csv_iteration_6_noisy_5000_40_train_ocrnet_miouweight,
    #         weight_func=miou)
    # get_csv(train_csv_iteration_6_noisy_5000_50_train, train_csv_iteration_6_noisy_5000_50_train_ocrnet_miouweight,
    #         weight_func=miou)
    #
    # mix_weight(train_csv_iteration_6_noisy_5000_10_train_ocrnet_bfweight, train_csv_iteration_6_noisy_5000_10_train_ocrnet_miouweight,
    #            train_csv_iteration_6_noisy_5000_10_train_ocrnet_miou_bf_weight)
    # mix_weight(train_csv_iteration_6_noisy_5000_20_train_ocrnet_bfweight, train_csv_iteration_6_noisy_5000_20_train_ocrnet_miouweight,
    #            train_csv_iteration_6_noisy_5000_20_train_ocrnet_miou_bf_weight)
    # mix_weight(train_csv_iteration_6_noisy_5000_30_train_ocrnet_bfweight, train_csv_iteration_6_noisy_5000_30_train_ocrnet_miouweight,
    #            train_csv_iteration_6_noisy_5000_30_train_ocrnet_miou_bf_weight)
    # mix_weight(train_csv_iteration_6_noisy_5000_40_train_ocrnet_bfweight, train_csv_iteration_6_noisy_5000_40_train_ocrnet_miouweight,
    #            train_csv_iteration_6_noisy_5000_40_train_ocrnet_miou_bf_weight)
    # mix_weight(train_csv_iteration_6_noisy_5000_50_train_bfweight, train_csv_iteration_6_noisy_5000_50_train_miouweight,
    #            train_csv_iteration_6_noisy_5000_50_train_ocrnet_miou_bf_weight)

    # get_csv(train_csv_iteration_6_noisy_5000_10_train, train_csv_iteration_6_noisy_5000_10_train_bfweight,
    #         weight_func=bfscore)
    # get_csv(train_csv_iteration_6_noisy_5000_20_train, train_csv_iteration_6_noisy_5000_20_train_bfweight,
    #         weight_func=bfscore)
    # get_csv(train_csv_iteration_6_noisy_5000_30_train, train_csv_iteration_6_noisy_5000_30_train_bfweight,
    #         weight_func=bfscore)
    # get_csv(train_csv_iteration_6_noisy_5000_40_train, train_csv_iteration_6_noisy_5000_40_train_bfweight,
    #         weight_func=bfscore)
    # get_csv(train_csv_iteration_6_noisy_5000_50_train, train_csv_iteration_6_noisy_5000_50_train_bfweight,
    #         weight_func=bfscore)

    # get_csv(train_csv_iteration_6_noisy_5000_10_train, train_csv_iteration_6_noisy_5000_10_train_diceweight,
    #         weight_func=dice)
    # get_csv(train_csv_iteration_6_noisy_5000_20_train, train_csv_iteration_6_noisy_5000_20_train_diceweight,
    #         weight_func=dice)
    # get_csv(train_csv_iteration_6_noisy_5000_30_train, train_csv_iteration_6_noisy_5000_30_train_diceweight,
    #         weight_func=dice)
    # get_csv(train_csv_iteration_6_noisy_5000_40_train, train_csv_iteration_6_noisy_5000_40_train_diceweight,
    #         weight_func=dice)
    # get_csv(train_csv_iteration_6_noisy_5000_50_train, train_csv_iteration_6_noisy_5000_50_train_diceweight,
    #         weight_func=dice)

    # get_csv(train_csv_iteration_6_noisy_5000_10_train, train_csv_iteration_6_noisy_5000_10_train_miouweight,
    #         weight_func=miou)
    # get_csv(train_csv_iteration_6_noisy_5000_20_train, train_csv_iteration_6_noisy_5000_20_train_miouweight,
    #         weight_func=miou)
    # get_csv(train_csv_iteration_6_noisy_5000_30_train, train_csv_iteration_6_noisy_5000_30_train_miouweight,
    #         weight_func=miou)
    # get_csv(train_csv_iteration_6_noisy_5000_40_train, train_csv_iteration_6_noisy_5000_40_train_miouweight,
    #         weight_func=miou)
    # get_csv(train_csv_iteration_6_noisy_5000_50_train, train_csv_iteration_6_noisy_5000_50_train_miouweight,
    #         weight_func=miou)

    # get_csv(train_csv_iteration_6_noisy_5000_10_train, train_csv_iteration_6_noisy_5000_10_train_bjweight,
    #         weight_func=bjscore)
    # get_csv(train_csv_iteration_6_noisy_5000_20_train, train_csv_iteration_6_noisy_5000_20_train_bjweight,
    #         weight_func=bjscore)
    # get_csv(train_csv_iteration_6_noisy_5000_30_train, train_csv_iteration_6_noisy_5000_30_train_bjweight,
    #         weight_func=bjscore)
    # get_csv(train_csv_iteration_6_noisy_5000_40_train, train_csv_iteration_6_noisy_5000_40_train_bjweight,
    #         weight_func=bjscore)
    # get_csv(train_csv_iteration_6_noisy_5000_50_train, train_csv_iteration_6_noisy_5000_50_train_bjweight,
    #         weight_func=bjscore)

    # mix_weight(train_csv_iteration_6_noisy_5000_10_train_bfweight, train_csv_iteration_6_noisy_5000_10_train_miouweight,
    #            train_csv_iteration_6_noisy_5000_10_train_miou_bf_weight)
    # mix_weight(train_csv_iteration_6_noisy_5000_20_train_bfweight, train_csv_iteration_6_noisy_5000_20_train_miouweight,
    #            train_csv_iteration_6_noisy_5000_20_train_miou_bf_weight)
    # mix_weight(train_csv_iteration_6_noisy_5000_30_train_bfweight, train_csv_iteration_6_noisy_5000_30_train_miouweight,
    #            train_csv_iteration_6_noisy_5000_30_train_miou_bf_weight)
    # mix_weight(train_csv_iteration_6_noisy_5000_40_train_bfweight, train_csv_iteration_6_noisy_5000_40_train_miouweight,
    #            train_csv_iteration_6_noisy_5000_40_train_miou_bf_weight)
    # mix_weight(train_csv_iteration_6_noisy_5000_50_train_bfweight, train_csv_iteration_6_noisy_5000_50_train_miouweight,
    #            train_csv_iteration_6_noisy_5000_50_train_miou_bf_weight)
