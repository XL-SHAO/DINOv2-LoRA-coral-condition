import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
import imageio
import datasets.imutils as imutils
import torch
import json
import argparse
import albumentations as A
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import random

def img_loader(path):
    img = imageio.imread(path).astype(np.float32)
    return img


class CoralReefDataSet(Dataset):
    def __init__(self, dataset_path, data_info_path, max_iters=None, type='train', resize=512, data_loader=img_loader):
        self.dataset_path = dataset_path

        with open(data_info_path, 'r') as f:
            self.data_list = json.load(f)
            print('data list loaded successfully')
            # print(self.data_list)

        self.loader = data_loader
        self.type = type
        self.resize = resize

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        # print(self.data_list)
        random.shuffle(self.data_list)

    def __transforms(self, aug, img):
        if aug:
            # img = imutils.random_fliplr(img)
            # img = imutils.random_flipud(img)
            # img = imutils.random_rot(img)
            # image = self.color_jittor(image)
            transform = A.Compose([
                # A.HistogramMatching([trg_img], p=0.5, read_fn=lambda x: x), 
                A.Resize(height=504, width=504),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1)
            ])

        else:
            transform = A.Compose([
                A.Resize(height=self.resize, width=self.resize)
            ])

        transformed_data = transform(image=img)

        img = transformed_data["image"]
        img = np.asarray(img)
        img = imutils.normalize_img(img)  # imagenet normalization
        img = np.transpose(img, (2, 0, 1))  # pytorch requires channel, head, weight

        return img


    def __getitem__(self, index):
        entry = self.data_list[index]

        parts =  entry['patchid'].split("/")
        img_path = os.path.join(self.dataset_path, parts[0], 'img', parts[1] + '.jpg')
        

        img = self.loader(img_path)
        # img = cv.resize(img, dsize=(224, 224), interpolation=cv.INTER_AREA) # For ViT training

        labels = [value for key, value in entry.items() if key != 'patchid']
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.type == 'train':
            img = self.__transforms(True, img)
        else:
            img  = self.__transforms(False, img)
        return img, labels, entry['patchid']


    def __len__(self):
        return len(self.data_list) 


class CoralReefDataSet_HRNet(Dataset):
    def __init__(self, dataset_path, data_info_path, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path

        with open(data_info_path, 'r') as f:
            self.data_list = json.load(f)
            print('data list loaded successfully')
            # print(self.data_list)

        self.loader = data_loader
        self.type = type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        # print(self.data_list)
        random.shuffle(self.data_list)

    def __transforms(self, aug, img):
        if aug:
            # img = imutils.random_fliplr(img)
            # img = imutils.random_flipud(img)
            # img = imutils.random_rot(img)
            # image = self.color_jittor(image)
            transform = A.Compose([
                # A.HistogramMatching([trg_img], p=0.5, read_fn=lambda x: x), 
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1)
            ])

        else:
            transform = A.Compose([
                A.Resize(height=512, width=512)
            ])

        transformed_data = transform(image=img)

        img = transformed_data["image"]
        img = np.asarray(img)
        img = imutils.normalize_img(img)  # imagenet normalization
        img = np.transpose(img, (2, 0, 1))  # pytorch requires channel, head, weight

        return img


    def __getitem__(self, index):
        entry = self.data_list[index]

        parts =  entry['patchid'].split("/")
        img_path = os.path.join(self.dataset_path, parts[0], 'img', parts[1] + '.jpg')
        

        img = self.loader(img_path)
        # img = cv.resize(img, dsize=(224, 224), interpolation=cv.INTER_AREA) # For ViT training

        labels = [value for key, value in entry.items() if key != 'patchid']
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.type == 'train':
            img = self.__transforms(True, img)
        else:
            img  = self.__transforms(False, img)
        return img, labels, entry['patchid']


    def __len__(self):
        return len(self.data_list) 
    

class CoralReefDataSetSemiS(Dataset):
    def __init__(self, dataset_path, data_info_path, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        with open(data_info_path, 'r') as f:
            self.data_list = json.load(f)

        self.loader = data_loader
        self.type = type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]

    def __transforms(self, aug, img):
        if aug:
            # img = imutils.random_fliplr(img)
            # img = imutils.random_flipud(img)
            # img = imutils.random_rot(img)
            # image = self.color_jittor(image)
            transform_1 = A.Compose([
                # A.HistogramMatching([trg_img], p=0.5, read_fn=lambda x: x), 
                A.Resize(height=504, width=504),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1)
            ])

            transform_2 = A.Compose([
                # A.HistogramMatching([trg_img], p=0.5, read_fn=lambda x: x), 
                A.Resize(height=504, width=504),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1),
                # A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=1, var_limit=20)
            ])

        else:
            transform_1 = A.Compose([
                A.Resize(height=504, width=504)
            ])

            transform_2 = A.Compose([
                A.Resize(height=504, width=504)
            ])
            
        transformed_data_1 = transform_1(image=img)

        img_1 = transformed_data_1["image"]
        img_1 = np.asarray(img_1)
        img_1 = imutils.normalize_img(img_1)  # imagenet normalization
        img_1 = np.transpose(img_1, (2, 0, 1))  # pytorch requires channel, head, weight

        transformed_data_2 = transform_2(image=img)

        img_2 = transformed_data_2["image"]
        img_2 = np.asarray(img_2)
        img_2 = imutils.normalize_img(img_2)  # imagenet normalization
        img_2 = np.transpose(img_2, (2, 0, 1))  # pytorch requires channel, head, weight

        return img_1, img_2


    def __getitem__(self, index):
        entry = self.data_list[index]

        parts =  entry['patchid'].split("/")
        img_path = os.path.join(self.dataset_path, parts[0], 'img', parts[1] + '.jpg')
        

        img = self.loader(img_path)
        # img = cv.resize(img, dsize=(224, 224), interpolation=cv.INTER_AREA) # For ViT training

        labels = [value for key, value in entry.items() if key != 'patchid']
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.type == 'train':
            img, img_2 = self.__transforms(True, img)
        else:
            img, img_2  = self.__transforms(False, img)
        return img, img_2, labels, entry['patchid']

    def __len__(self):
        return len(self.data_list) 

class CoralReefDataSetHF(Dataset):
    def __init__(self, dataset_path, data_info_path, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        with open(data_info_path, 'r') as f:
            self.data_list = json.load(f)

        self.loader = data_loader
        self.type = type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]

    def __transforms(self, aug, img, hf_img):
        if aug:
            # img = imutils.random_fliplr(img)
            # img = imutils.random_flipud(img)
            # img = imutils.random_rot(img)
            # image = self.color_jittor(image)
            transform = A.Compose([
                # A.HistogramMatching([trg_img], p=0.5, read_fn=lambda x: x), 
                A.Resize(height=504, width=504),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1)
            ])

            
        else:
            transform = A.Compose([
                A.Resize(height=504, width=504)
            ])
            

        transformed_data = transform(image=img)

        img = transformed_data["image"]
        img = np.asarray(img)
        img = imutils.normalize_img(img)  # imagenet normalization
        img = np.transpose(img, (2, 0, 1))  # pytorch requires channel, head, weight


        transformed_hf_img = transform(image=hf_img)

        hf_img = transformed_hf_img["image"]
        hf_img = np.asarray(hf_img)
        hf_img = imutils.normalize_img(hf_img)  # imagenet normalization
        hf_img = np.transpose(hf_img, (2, 0, 1))  # pytorch requires channel, head, weight

        return img, hf_img


    def high_pass_filter(self, img):
        """Apply a high-frequency pass filter using Fourier transform and replicate to three channels."""
        # Convert to grayscale if it's not already
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # Perform Fourier transform
        f = fft2(img_gray)
        fshift = fftshift(f)

        # Create a high-pass filter mask
        rows, cols = img_gray.shape
        crow, ccol = rows // 2 , cols // 2  # Center of the image

        # Define a radius for the high-pass filter
        radius = 30
        mask = np.ones((rows, cols), np.uint8)
        mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0

        # Apply the mask
        fshift = fshift * mask

        # Inverse Fourier transform to get the image back
        f_ishift = ifftshift(fshift)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize the result
        img_high_freq = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Replicate the single-channel result to create a three-channel image
        img_high_freq_three_channels = np.stack([img_high_freq] * 3, axis=-1)

        return img_high_freq_three_channels

    def __getitem__(self, index):
        entry = self.data_list[index]

        parts =  entry['patchid'].split("/")
        img_path = os.path.join(self.dataset_path, parts[0], 'img', parts[1] + '.jpg')
        

        img = self.loader(img_path)
        fre_img = self.high_pass_filter(img)
        # img = cv.resize(img, dsize=(224, 224), interpolation=cv.INTER_AREA) # For ViT training

        labels = [value for key, value in entry.items() if key != 'patchid']
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.type == 'train':
            img_1, img_2  = self.__transforms(True, img, fre_img)
        else:
            img_1, img_2  = self.__transforms(False, img, fre_img)
        return img_1, img_2, labels, entry['patchid']

    def __len__(self):
        return len(self.data_list) 


class CoralReefDataSetTTA(Dataset):
    def __init__(self, dataset_path, data_info_path, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        with open(data_info_path, 'r') as f:
            self.data_list = json.load(f)

        self.loader = data_loader
        self.type = type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]

    def __transforms(self, aug, img):
        if aug:
            # img = imutils.random_fliplr(img)
            # img = imutils.random_flipud(img)
            # img = imutils.random_rot(img)
            # image = self.color_jittor(image)
            transform_1 = A.Compose([
                # A.HistogramMatching([trg_img], p=0.5, read_fn=lambda x: x), 
                A.Resize(height=504, width=504),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1),
                # A.RandomBrightnessContrast(p=0.5),
                # A.GaussNoise(p=0.5, var_limit=15)
            ])

            transform_2 = A.Compose([
                # A.HistogramMatching([trg_img], p=0.5, read_fn=lambda x: x), 
                A.Resize(height=504, width=504),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1),
                # A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=1, var_limit=20)
            ])
        else:
            transform_1 = A.Compose([
                A.Resize(height=504, width=504)
            ])
            transform_2 = A.Compose([
                A.Resize(height=504, width=504)
            ])

        transformed_data_1 = transform_1(image=img)
        transformed_data_2 = transform_2(image=img)

        img_1 = transformed_data_1["image"]
        img_1 = np.asarray(img_1)
        img_1 = imutils.normalize_img(img_1)  # imagenet normalization
        img_1 = np.transpose(img_1, (2, 0, 1))  # pytorch requires channel, head, weight

        img_2 = transformed_data_2["image"]
        img_2 = np.asarray(img_2)
        img_2 = imutils.normalize_img(img_2)  # imagenet normalization
        img_2 = np.transpose(img_2, (2, 0, 1))  # pytorch requires channel, head, weight
        return img_1, img_2

    def __getitem__(self, index):
        entry = self.data_list[index]

        parts =  entry['patchid']
        img_path = os.path.join(self.dataset_path, parts + '.jpg')
        

        img = self.loader(img_path)
        # img = cv.resize(img, dsize=(224, 224), interpolation=cv.INTER_AREA) # For ViT training

        labels = [value for key, value in entry.items() if key != 'patchid']
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.type == 'train':
            img_1, img_2 = self.__transforms(True, img)
        else:
            img_1, img_2 = self.__transforms(False, img)
        return img_1, img_2, labels, entry['patchid']

    def __len__(self):
        return len(self.data_list) 


class CoralReefDataSetPL(Dataset):
    def __init__(self, dataset_path, data_info_path, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        with open(data_info_path, 'r') as f:
            self.data_list = json.load(f)

        self.loader = data_loader
        self.type = type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]

    def __transforms(self, aug, img):
        if aug:
            # img = imutils.random_fliplr(img)
            # img = imutils.random_flipud(img)
            # img = imutils.random_rot(img)
            # image = self.color_jittor(image)
            transform = A.Compose([
                # A.HistogramMatching([trg_img], p=0.5, read_fn=lambda x: x), 
                A.Resize(height=504, width=504),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1)
            ])
        else:
            transform = A.Compose([
                A.Resize(height=504, width=504),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1)
            ])

        transformed = transform(image=img)
        img = transformed["image"]
        img = np.asarray(img)
        img = imutils.normalize_img(img)  # imagenet normalization
        img = np.transpose(img, (2, 0, 1))  # pytorch requires channel, head, weight
        return img

    def __getitem__(self, index):
        entry = self.data_list[index]

        parts =  entry['patchid']
        img_path = os.path.join(self.dataset_path, parts + '.jpg')
        

        img = self.loader(img_path)
        # img = cv.resize(img, dsize=(224, 224), interpolation=cv.INTER_AREA) # For ViT training

        labels = [value for key, value in entry.items() if key != 'patchid']
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.type == 'train':
            img = self.__transforms(True, img)
        else:
            img = self.__transforms(False, img)
        return img, labels, entry['patchid']

    def __len__(self):
        return len(self.data_list) 

class CoralReefDomainAdaptationDataSet(Dataset):
    def __init__(self, dataset_path, source_data_info_path, target_data_info_path, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path

        with open(source_data_info_path, 'r') as f:
            self.source_data_info_path = json.load(f)
        with open(target_data_info_path, 'r') as f:
            self.target_data_info_path = json.load(f)

        self.loader = data_loader
        self.type = type

        if max_iters is not None:
            self.source_data_info_path = self.source_data_info_path * int(np.ceil(float(max_iters) / len(self.source_data_info_path)))
            self.source_data_info_path = self.source_data_info_path[0:max_iters]

            self.target_data_info_path = self.target_data_info_path * int(np.ceil(float(max_iters) / len(self.target_data_info_path)))
            self.target_data_info_path = self.target_data_info_path[0:max_iters]

    def __transforms(self, aug, src_img, trg_img):
        if aug:
            # img = imutils.random_fliplr(img)
            # img = imutils.random_flipud(img)
            # img = imutils.random_rot(img)
            # image = self.color_jittor(image)
            transform = A.Compose([
                A.HistogramMatching([trg_img], p=0.5, read_fn=lambda x: x), 
                A.Resize(height=504, width=504),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1),
            ])
        else:
            transform = A.Compose([
                A.Resize(height=504, width=504),
            ])
        transformed = transform(image=src_img)
        src_img = transformed["image"]

        src_img = np.asarray(src_img)
        src_img = imutils.normalize_img(src_img)  # imagenet normalization
        src_img = np.transpose(src_img, (2, 0, 1))  # pytorch requires channel, head, weight
        return src_img, trg_img

    def __getitem__(self, index):
        source_entry = self.source_data_info_path[index]
        target_entry = self.target_data_info_path[index]

        source_parts =  source_entry['patchid'].split("/")
        target_parts =  target_entry['patchid'].split("/")

        source_img_path = os.path.join(self.dataset_path, source_parts[0], 'img', source_parts[1] + '.jpg')
        target_img_path = os.path.join(self.dataset_path, target_parts[0], 'img', target_parts[1] + '.jpg')

        source_img = self.loader(source_img_path)
        target_img = self.loader(target_img_path)

        # img = cv.resize(img, dsize=(224, 224), interpolation=cv.INTER_AREA) # For ViT training

        source_labels = [value for key, value in source_entry.items() if key != 'patchid']
        source_labels = torch.tensor(source_labels, dtype=torch.float32)

        if self.type == 'train':
            source_img, target_img = self.__transforms(True, source_img, target_img)
        else:
            source_img, target_img = self.__transforms(False, source_img, target_img)
        return source_img, target_img, source_labels, source_entry['patchid'], target_entry['patchid']

    def __len__(self):
        return len(self.target_data_info_path)

def make_data_loader(
        args):  # **kwargs was the second argument and was omitted (to be tested) note that it was also 4th argument of the
    # DataLoader and it was omitted (to be tested)
    if args.dataset == 'coral_reef':
        # Creating a torch.utils.data.Dataset ready for the torch.utils.data.DataLoader
        dataset = CoralReefDataSet(args.dataset_path, args.data_info_path, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8,
                                 drop_last=False)
        # set num_workers to 4 because of potential slowing down with 16 on CERN GPU
        # drop_last (bool, optional) – set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
        # If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)
        return data_loader

    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coral Reef DataLoader Test")
    parser.add_argument('--dataset', type=str, default='coral_reef')
    parser.add_argument('--max_iters', type=int)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/coralReef/data/test_img')
    parser.add_argument('--data_info_path', type=str,
                        default='D:/Workspace/Python/coralReef/data/test_img/train_set.json')
    parser.add_argument('--shuffle', type=bool,
                        default=True)  # shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_name_list', type=list)
    args = parser.parse_args()
    # Reading from data_list_path which is set as default to './xBD_list/train.txt'
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        img, label, _ = data
        print(i, "个inputs", img.data.size())
