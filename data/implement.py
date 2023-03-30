import torch
import logging
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class BasicDataset_without_weight(Dataset):
    def __init__(self, file_csv, transform):
        super(BasicDataset_without_weight, self).__init__()
        self.df = pd.read_csv(file_csv)
        self.transform = transform
        logging.info(f'Creating dataset with {len(self.df)} examples')


    def __getitem__(self, index):
        img = Image.open(self.df.iloc[index, 0]).convert('RGB')
        img = self.transform(img)

        mask = Image.open(self.df.iloc[index, 1]).convert('L')
        mask = self.transform(mask)

        return {
            'image': img,
            'mask': mask,
            'name': self.df.iloc[index, 0],
        }

    def __len__(self):
        return len(self.df)


class BasicDataset_with_weight(Dataset):
    def __init__(self, file_csv, transform):
        super(BasicDataset_with_weight, self).__init__()
        self.df = pd.read_csv(file_csv)
        self.transform = transform
        logging.info(f'Creating dataset with {len(self.df)} examples')

    def __getitem__(self, index):
        img = Image.open(self.df.iloc[index, 0]).convert('RGB')
        img = self.transform(img)

        mask = Image.open(self.df.iloc[index, 1]).convert('L')
        mask = self.transform(mask)

        weight = self.df.iloc[index, 2]

        return {
            'image': img,
            'mask': mask,
            'weight': torch.from_numpy(np.array(weight)),
            'name': self.df.iloc[index, 0]
        }

    def __len__(self):
        return len(self.df)


train_transform = transforms.Compose([
    transforms.Resize([321, 321]),
    transforms.ToTensor(),
])

train_transform_deeplabv3 = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
])


train_transform_cityscapes = transforms.Compose([
    transforms.Resize([256, 256]),
    # transforms.RandomCrop(size=(512, 512)),
    transforms.ToTensor(),
])

if __name__ == '__main__':
    dir_img = "data/supervisely/train/"
    dir_mask = "data/supervisely/train_mask/"
    scale = 1.0
    val_percent = 0.1
    batch_size = 8

    train_dataset = BasicDataset_without_weight(file_csv='data/csv/train_noisy_60.csv',
                                                transform=train_transform)
    valid_dataset = BasicDataset_without_weight(file_csv="data/csv/valid_noisy_60.csv",
                                                transform=train_transform)
    test_dataset = BasicDataset_without_weight(file_csv="data/csv/test.csv",
                                               transform=train_transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))

    for batch in train_dataloader:
        imgs = batch['image']
        true_masks = batch['mask']
        print("单个img的size: ", imgs.shape)
        print("单个mask的size: ", true_masks.shape)
        assert imgs.shape[1] == 3, \
            f'Network has been defined with 3 input channels, ' \
            f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'
