import os
import math
import copy
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from utils.add_noise import add_noise_iteration
from sklearn.model_selection import train_test_split
from config.noisy_dataset_path import *

mask_path = "data/Supervisely/train_mask"
csv_path = "data/Supervisely/csv"

mask_path_cls = "data/Supervisely/train_mask_cls"
csv_path_cls = "data/Supervisely/csv_cls"


def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [sorted(arr[i:i + n]) for i in range(0, len(arr), n)]


# split dataset and add label_noise
def split_dataset(dir_noisy_mask, dir_train_csv, dir_valid_csv, ratio, iterations):
    # count 记录有多少样本加了噪声，noise_dict记录对应图片所加的噪声等级
    count = 0
    noise_dict = {}

    # id_add_noise 是需要加噪声的id列表，id_list是根据传进来的iterations列表将其分为几组。例如传进来【6,10】，则会将id_add_noise分为5组
    ids_shuffle = copy.deepcopy(train_test_id)
    id_add_noise = random.sample(ids_shuffle, int(ratio * len(ids_shuffle)))
    id_list = chunks(id_add_noise, iterations[1] - iterations[0] + 1)

    if not os.path.exists(dir_noisy_mask):
        os.mkdir(dir_noisy_mask)

    for id in tqdm(train_test_id):
        mask_file = os.path.join(mask_path_all, f'{id}.png')
        mask = cv2.imread(os.path.join(mask_path_all, mask_file), cv2.IMREAD_GRAYSCALE)

        # iteration是指迭代次数，id_iteration是当前迭代次数的id列表
        for iteration, id_iteration in enumerate(id_list):
            if id in id_iteration:
                count += 1
                iteration_now = iterations[0] + iteration
                mask = add_noise_iteration(mask, iterations=iteration_now)
                noise_dict[os.path.join(dir_noisy_mask, f"{id}.png")] = iteration_now
                plt.imsave(fname=os.path.join(dir_noisy_mask, f"{id}.png"), arr=mask, cmap=cm.gray)
                break

    print(f'添加噪声完成，noisy count is {count}!')
    get_csv(img_dir=img_path_all, mask_dir=dir_noisy_mask, csv_train_path=dir_train_csv, csv_valid_path=dir_valid_csv,
            noise_dict=noise_dict, id_add_noise=id_add_noise)


def get_csv(img_dir, mask_dir, csv_train_path, csv_valid_path, noise_dict, id_add_noise):
    train_dict = {}
    valid_dict = {}
    all_img_list = []

    for id in tqdm(train_test_id):
        img_file = os.path.join(img_dir, f'{id}.png')
        all_img_list.append(img_file)

    train_list, valid_list = train_test_split(all_img_list, test_size=0.1, random_state=2020)

    for item in tqdm(train_list):
        id = os.path.splitext(item.split('/')[-1])[0]
        if id not in id_add_noise:
            mask_file = os.path.join(mask_path_all, f'{id}.png')
            train_dict[item] = [mask_file, 0]
        else:
            mask_file = os.path.join(mask_dir, f'{id}.png')
            train_dict[item] = [mask_file, noise_dict[mask_file]]

    for item in tqdm(valid_list):
        id = os.path.splitext(item.split('/')[-1])[0]
        if id not in id_add_noise:
            mask_file = os.path.join(mask_path_all, f'{id}.png')
            valid_dict[item] = [mask_file, 0]
        else:
            mask_file = os.path.join(mask_dir, f'{id}.png')
            valid_dict[item] = [mask_file, noise_dict[mask_file]]

    train_df = pd.DataFrame.from_dict(train_dict, orient='index')
    valid_df = pd.DataFrame.from_dict(valid_dict, orient='index')

    train_df.to_csv(csv_train_path)
    valid_df.to_csv(csv_valid_path)
    print('--生成 csv 完成！')


#
# def split_dataset_Noise_cls(mask_path_save):
#     # 遍历每张图像，计算不同的噪声等级下的mask，然后给出不同的等级标签，iteration为0-9, 标签分数为10-1
#     if not os.path.exists(mask_path_save):
#         os.mkdir(mask_path_save)
#     ids_shuffle = copy.deepcopy(ids)
#     random.shuffle(ids_shuffle)
#     id_list = chunks(ids_shuffle, 10)
#     noise_dict = {}
#
#     for id in tqdm(ids):
#         mask_file = glob(os.path.join(mask_path_all, id) + '.*')
#         mask = cv2.imread(os.path.join(mask_path_all, mask_file[0]), cv2.IMREAD_GRAYSCALE)
#         for index, list_item in enumerate(id_list):
#             if id in list_item:
#                 # if index != 0:
#                 #     mask = add_noise_iteration(mask, iterations=index)
#                 noise_dict[os.path.join(mask_path_save, f"{id}.png")] = 9 - index
#                 # print(os.path.join(mask_path_save, f"{id}.png"))
#                 # plt.imsave(fname=os.path.join(mask_path_save, f"{id}.png"), arr=mask, cmap=cm.gray)
#                 break
#     return noise_dict


# def get_csv_noise_cls(img_dir, mask_dir, csv_path_save, noise_dict):
#     if not os.path.exists(csv_path_save):
#         os.mkdir(csv_path_save)
#     all_img_list = []
#     for id in tqdm(ids):
#         img_file = glob(os.path.join(img_dir, id) + '.*')
#         img_file = os.path.join(img_dir, img_file[0])
#         all_img_list.append(img_file)
#     train_list, valid_test_list = train_test_split(all_img_list, test_size=0.2, random_state=2020)
#     valid_list, test_list = train_test_split(valid_test_list, test_size=0.5, random_state=2020)
#
#     train_dict = {}
#     valid_dict = {}
#     test_dict = {}
#     for item in tqdm(train_list):
#         id = os.path.splitext(item.split('/')[-1])[0]
#         mask_file = glob(os.path.join(mask_dir, id) + '.*')
#         mask_file = os.path.join(mask_dir, mask_file[0])
#         train_dict[item] = [mask_file, noise_dict[mask_file]]
#
#     for item in tqdm(valid_list):
#         id = os.path.splitext(item.split('/')[-1])[0]
#         mask_file = glob(os.path.join(mask_dir, id) + '.*')
#         mask_file = os.path.join(mask_dir, mask_file[0])
#         valid_dict[item] = [mask_file, noise_dict[mask_file]]
#
#     for item in tqdm(test_list):
#         id = os.path.splitext(item.split('/')[-1])[0]
#         mask_file = glob(os.path.join(mask_dir, id) + '.*')
#         mask_file = os.path.join(mask_dir, mask_file[0])
#         test_dict[item] = [mask_file, noise_dict[mask_file]]
#
#     train_df = pd.DataFrame.from_dict(train_dict, orient='index')
#     valid_df = pd.DataFrame.from_dict(valid_dict, orient='index')
#     test_df = pd.DataFrame.from_dict(test_dict, orient='index')
#
#     train_df.to_csv(os.path.join(csv_path_save, "train.csv"))
#     valid_df.to_csv(os.path.join(csv_path_save, "valid.csv"))
#     test_df.to_csv(os.path.join(csv_path_save, "test.csv"))


if __name__ == '__main__':
    random.seed(2020)
    ids = sorted(
        [os.path.splitext(filename)[0] for filename in os.listdir(img_path_all) if not filename.startswith('.')])
    test_ids = sorted(random.sample(ids, 711))
    train_test_id = sorted(list(set(ids).difference(set(test_ids))))

    # # 生成test_csv
    # test_dict = {}
    # for id in tqdm(test_ids):
    #     img_file = os.path.join(img_path_all, f'{id}.png')
    #     mask_file = os.path.join(mask_path_all, f'{id}.png')
    #     test_dict[img_file] = mask_file
    # test_df = pd.DataFrame.from_dict(test_dict, orient='index')
    # test_df.to_csv(csv_test)
    # print('--生成test_csv完成')
    #
    # split_dataset(img_dir_iteration_6_noisy_5000_10, train_csv_iteration_6_noisy_5000_10_train,
    #               train_csv_iteration_6_noisy_5000_10_valid, ratio=0.1, iterations=[6, 10])
    # split_dataset(img_dir_iteration_6_noisy_5000_20, train_csv_iteration_6_noisy_5000_20_train,
    #               train_csv_iteration_6_noisy_5000_20_valid, ratio=0.2, iterations=[6, 10])
    # split_dataset(img_dir_iteration_6_noisy_5000_30, train_csv_iteration_6_noisy_5000_30_train,
    #               train_csv_iteration_6_noisy_5000_30_valid, ratio=0.3, iterations=[6, 10])
    # split_dataset(img_dir_iteration_6_noisy_5000_40, train_csv_iteration_6_noisy_5000_40_train,
    #               train_csv_iteration_6_noisy_5000_40_valid, ratio=0.4, iterations=[6, 10])
    # split_dataset(img_dir_iteration_6_noisy_5000_50, train_csv_iteration_6_noisy_5000_50_train,
    #               train_csv_iteration_6_noisy_5000_50_valid, ratio=0.5, iterations=[6, 10])

    get_csv(img_path_all, mask_path_all, csv_train_clean, csv_valid_clean, {}, [])
    # get_csv(img_path_all, mask_path_25, csv_path_25)
    # get_csv(img_path_all, mask_path_50, csv_path_50)
    # get_csv(img_path_all, mask_path_75, csv_path_75)
    # get_csv(img_path_all, mask_path_100, csv_path_100)

    # noise_dict = split_dataset_Noise_cls(mask_path_save=mask_path_cls)
    # get_csv_noise_cls(img_path_all, mask_path_cls, csv_path_cls, noise_dict)
