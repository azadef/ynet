#!/usr/bin/python
#
# Copyright 2022 Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


from typing import Callable
import os
import scipy.io as sio
from tqdm import tqdm
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from octprocessing import get_valid_img_seg_reimpl, get_unlabelled_bscans


def plot_label(path, label):
    x = np.load(path, allow_pickle=True).item()
    plt.imshow(x[label])
    plt.show()


def plot_img(path):
    plot_label(path, "images")


def plot_full_img(path, b_scan_idx=0):
    x = np.load(path, allow_pickle=True).item()
    fig, axes = plt.subplots(1,2)
    img, mask = get_valid_img_seg_reimpl(x)
    
    axes[0].imshow(mask[:,:,b_scan_idx], cmap=plt.cm.jet, vmax=9)
    axes[0].set_title("mask")
    axes[1].imshow(img[:,:,b_scan_idx])
    axes[1].set_title("image")
    plt.show()


def plot_sliced_img(dataset_path,sample_file_name):
    fig, axes = plt.subplots(1,2)
    img = np.load(os.path.join(os.path.join(dataset_path, "images"), sample_file_name))
    mask = np.load(os.path.join(os.path.join(dataset_path, "masks"), sample_file_name))
    axes[0].imshow(mask, cmap=plt.cm.jet, vmax=9)
    axes[0].set_title("mask")
    axes[1].imshow(img)
    axes[1].set_title("image")
    plt.show()


def test_slicing(path):
    obj = np.load(path, allow_pickle=True).item()
    x,_=get_valid_img_seg_reimpl(obj)
    x = torch.Tensor(x)
    slices = slicing(x)
    col_num = x.shape[1] // 12
    for b in range(x.shape[2]):
        for col in range(col_num):
            assert torch.all(x[:,col*12:(col + 1)*12,b] == slices[b*col_num + col]), f"slice {col} in image {b} is off"
    sum_scans = sum([e.shape[1] for e in slices])
    assert x.shape[1] // 12 * 12 * x.shape[2] == sum_scans , f"The number of a scans in samples {sum_scans} and the number of a"\
        + f" scans in complete file floored to col_width (# ascans per sample) {x.shape[1] // 12 * 12 * x.shape[2]} differ"
    

def pad_to_max_num(i:int, mx:int): # base 10
    length = int(np.ceil(np.log10(mx)))
    s = str(i)
    zeros = length - len(s)
    if zeros > 0:
        s = "".join([*["0" for i in range(zeros)],s])
    return s


def slice_to_bscans(x)->list:
    """
    input x (Duke): (H x W x B) (496, 768,61), B = 61 ~ 61 B-scans (HxW grayscale)
    input x (UMN): (H x W x B) (496, 1024,25), B = 25 ~ 25 B-scans (HxW grayscale)
     - note that dims given are dims of images yet the generated masks only have 11 B scans as rest not labelled
        and the # of a scans labelled per patient differs - leading to differing width   
    -> list of B slices H x W 
    """
    #  'images', 'automaticFluidDME', 'manualFluid1', 'manualFluid2' torch.Size([496, 768, 61])
    #  'automaticLayersDME', 'automaticLayersNormal', 'manualLayers1', 'manualLayers2', torch.Size([8, 768, 61])
    images = []

    for i in range(x.shape[2]):
        images.append(x[:,:,i])
    return images 


def slicing(x)->list:
    """
    input x: (H x W x B) (496,, 768,61), B = 61 ~ 61 B-scans (HxW grayscale)
    input x (UMN): (H x W x B) (496, 1024,25), B = 25 ~ 25 B-scans (HxW grayscale)
     - note that dims given are dims of images yet the generated masks only have 11 B scans as rest not labelled 
        and the # of a scans labelled per patient differs - leading to differing width   
    -> list of slices
    (i) slicing H x W x B into B separate H x W tensors ii) slice along H dimension to create columns)   
    """
    #  'images', 'automaticFluidDME', 'manualFluid1', 'manualFluid2' torch.Size([496, 768, 61])
    #  'automaticLayersDME', 'automaticLayersNormal', 'manualLayers1', 'manualLayers2', torch.Size([8, 768, 61])
    images = []

    for i in range(x.shape[2]):
        images.append(x[:,:,i])
    # 768 = 2**8 * 3  - for column-width=3* 2**i  i= 0..8 evenly divisible 
    # note the number of a scans differ by file as preprocessing of the mask is restricted as the A scan
    #     (column) range of labelling differs between images
    col_width=12 # # a scans per sample
    samples = []
    for img in images:
        for i in range(img.shape[1] // col_width):
            samples.append(img[:,i*col_width:(i+1)*col_width])
    return samples


class DataPreprocessor():
    """
    DataPreprocessor for loading Chiu 2015 oct dataset for image segmentation 
    image and corresponding mask are stored in a dictionary 

    init args:

    slicing: torch.Tensor -> [torch.Tensor] - slices a single image into multiple samples 

    slicing of oct images and masks into narrow columns (2d, grayscale)
    acc to author the Chiu 2015 oct dataset is composed of volumetrics scans: 61 B-scans with each being composed of 768 A-scans
    note:  an A-scan is a dx1 vector of pixels
    input x: (H x W x B) ({496,8}, 768,61), B = 61 ~ 61 B-scans (HxW grayscale), {496, 8} ~ either 496 or 8 depending on data  
    -> list of slices 
    (i) slicing H x W x B into B separate H x W tensors ii) slice along H dimension to create columns)


    preprocess():
    saves all slices as {'label_h':slicing(data_h)[j]}, i=1..len(self.labels), j= i%self.samples_per_file, h = i // self.samples_per_file
    to filename_{slice_num}.npy

    generates valid img and mask via octprocessing.get_valid_img_seg_reimpl()
    """
    def __init__(self, data_path, dest_path, slicing: Callable[[torch.Tensor], list], is_mat=False, labelled_dataset=True):
    

        self.ft_mat = { True:".mat", False:".npy" }
        self.labelled_dataset = labelled_dataset
        self.img_files = [os.path.abspath(e.path) for e in os.scandir(data_path) if e.is_file() 
                     and self.ft_mat[is_mat].lstrip(".") == e.name.split(".")[-1]]   
        assert len(self.img_files) > 0, f"no such {self.ft_mat[is_mat]} file types found in {data_path}"
        self.dest_path = dest_path
        assert data_path != dest_path, "Please provide a dest_path different from data_path" 
        self.slicing = slicing
        self.is_mat = is_mat
        self.img_slice_nums = {}     
        #determine samples_per_file
        for f in self.img_files:
            if is_mat:
                obj = sio.loadmat(f)
            else:
                obj = np.load(f, allow_pickle=True).item()
            
            # setting num slices and checks for labelled and unlabelled dataset
            if labelled_dataset:
                img, mask = get_valid_img_seg_reimpl(obj)
                img, mask = torch.Tensor(img), torch.Tensor(mask)
            
                self.img_slice_nums[f] = len(slicing(img))
                assert self.img_slice_nums[f] == len(slicing(mask)), f"num slices of image: {f}"\
                    + "  must equal the num slices of corresponding mask"
            else:
                img = get_unlabelled_bscans(obj)
                img = torch.Tensor(img)
                self.img_slice_nums[f] = len(slicing(img))
        #labels
        if labelled_dataset: 
            self.labels = set(["images", "masks"])
        else:
            self.labels = set(["images"])
        #create paths
        splits = ["train", "val", "test"]
        for lbl in self.labels:
            for sp in splits:
                path = os.path.join(self.dest_path, sp, lbl)
                if not os.path.isdir(path):
                    os.makedirs(path)

        super().__init__()

    def preprocess(self):
        print("Starting preprocessing:")
        for img_num in range(len(self.img_files)):
            print(f"pre-processing img {img_num +1}/{len(self.img_files)}:{self.img_files[img_num]}")
            split_path = ""
            subject_num = int(self.img_files[img_num].split("/")[-1].split(".")[0].split("_")[1])
            if subject_num < 7:
                split_path = "train"
            elif subject_num == 7 or subject_num == 8:
                split_path = "val"
            elif subject_num == 9 or subject_num == 10:
                split_path = "test"
            else:
                print("wrong subject number")
                assert False
            
            if self.is_mat:
                scan_obj = sio.loadmat(self.img_files[img_num])
            else: 
                scan_obj = np.load(self.img_files[img_num], allow_pickle=True).item()

            # extract labelled or unlabelled bscans
            if self.labelled_dataset:
                img, mask = get_valid_img_seg_reimpl(scan_obj)
                img, mask = torch.Tensor(img), torch.Tensor(mask)
                obj = {"images": img, "masks": mask}
            else:
                img = get_unlabelled_bscans(scan_obj)
                img = torch.Tensor(img)
                obj = {"images": img}

            slices = {}
            # convert and slice images and masks
            print(">>> slicing")
            for k in tqdm(obj.keys()):
                if k in self.labels:
                    slices[k] = self.slicing(obj[k])
            # rearrange into img slice, mask slice tuples and  add slices to buffer
            print(">>> saving")
            total_slices = self.img_slice_nums[self.img_files[img_num]]
            for slice_num in tqdm(range(total_slices)):
                for k in self.labels:
                    sample = slices[k][slice_num]
                    #print(sample.shape)
                    label_path = os.path.join(self.dest_path, split_path, k)
                    fname = f"{self.img_files[img_num].replace(self.ft_mat[self.is_mat], '')}_{pad_to_max_num(slice_num, total_slices)}.npy"
                    np.save(os.path.join(label_path, os.path.basename(fname)), sample)


class DataPreprocessorUMN():
    """
    DataPreprocessor for loading Chiu 2015 oct dataset for image segmentation
    image and corresponding mask are stored in a dictionary

    init args:

    slicing: torch.Tensor -> [torch.Tensor] - slices a single image into multiple samples

    slicing of oct images and masks into narrow columns (2d, grayscale)
    acc to author the Chiu 2015 oct dataset is composed of volumetrics scans: 61 B-scans with each being composed of 768 A-scans
    note:  an A-scan is a dx1 vector of pixels
    input x: (H x W x B) ({496,8}, 768,61), B = 61 ~ 61 B-scans (HxW grayscale), {496, 8} ~ either 496 or 8 depending on data
    -> list of slices
    (i) slicing H x W x B into B separate H x W tensors ii) slice along H dimension to create columns)


    preprocess():
    saves all slices as {'label_h':slicing(data_h)[j]}, i=1..len(self.labels), j= i%self.samples_per_file, h = i // self.samples_per_file
    to filename_{slice_num}.npy

    generates valid img and mask via octprocessing.get_valid_img_seg_reimpl()
    """

    def __init__(self, data_path, dest_path, slicing: Callable[[torch.Tensor], list], is_mat=False):

        self.ft_mat = True
        self.labelled_dataset = True
        self.dataset = sio.loadmat(data_path)
        self.images = self.dataset['AllSubjects'][0][:29]
        self.masks = self.dataset['ManualFluid1'][0]

        self.dest_path = dest_path
        assert data_path != dest_path, "Please provide a dest_path different from data_path"

        self.slicing = slicing
        self.is_mat = is_mat
        self.img_slice_nums = {}
        # determine samples_per_file
        for i in range(29):
            img, mask = self.images[i], self.masks[i]
            img, mask = torch.Tensor(img), torch.Tensor(mask)

            self.img_slice_nums[str(i)] = len(slicing(img))
            assert self.img_slice_nums[str(i)] == len(slicing(mask)), f"num slices of image: {f}" \
                                                                 + "  must equal the num slices of corresponding mask"

        # labels
        self.labels = set(["images", "masks"])
        # create paths
        splits = ["train", "val", "test"]
        for lbl in self.labels:
            for sp in splits:
                path = os.path.join(self.dest_path, sp, lbl)
                if not os.path.isdir(path):
                    os.makedirs(path)

        super().__init__()

    def preprocess(self):
        print("Starting preprocessing:")
        for img_num in range(len(self.images)):
            print(f"pre-processing img {img_num + 1}/{len(self.images)}")
            split_path = ""
            subject_num = img_num
            if subject_num < 19:
                split_path = "train"
            elif subject_num >= 19 and subject_num < 24:
                split_path = "val"
            elif subject_num >= 24:
                split_path = "test"
            else:
                print("wrong subject number")
                assert False

            img, mask = self.images[img_num], self.masks[img_num]
            # extract labelled or unlabelled bscans

            img, mask = torch.Tensor(img), torch.Tensor(mask)
            obj = {"images": img, "masks": mask}

            slices = {}
            # convert and slice images and masks
            print(">>> slicing")
            for k in tqdm(obj.keys()):
                if k in self.labels:
                    slices[k] = self.slicing(obj[k])
            # rearrange into img slice, mask slice tuples and  add slices to buffer
            print(">>> saving")
            total_slices = self.img_slice_nums[str(img_num)]
            for slice_num in tqdm(range(total_slices)):
                for k in self.labels:
                    sample = slices[k][slice_num]
                    # print(sample.shape)
                    label_path = os.path.join(self.dest_path, split_path, k)
                    fname = f"{str(img_num)}_{pad_to_max_num(slice_num, total_slices)}.npy"
                    np.save(os.path.join(label_path, os.path.basename(fname)), sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing script for oct dataset 2015")
    arg_path = parser.add_argument("data_path", metavar="path",type=str)
    parser.add_argument("dest_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--is_mat", type=bool, default=True)
    parser.add_argument("--full_bscan", type=bool, default=True)
    parser.add_argument("--extract_dataset", choices=["labelled_dataset", "unlabelled_dataset"], default="labelled_dataset")
    args = parser.parse_args()

    labelled_dataset = args.extract_dataset == "labelled_dataset"
    
    if not os.path.isfile(args.data_path):
        argparse.ArgumentError(arg_path, "'data_path' must be a file")
    if args.dataset == "UMN":
        processor = DataPreprocessorUMN(args.data_path, args.dest_path, slice_to_bscans, is_mat=args.is_mat)
    elif args.full_bscan:
        processor = DataPreprocessor(args.data_path, args.dest_path, slice_to_bscans, is_mat=args.is_mat, labelled_dataset=labelled_dataset)
    else:
        processor = DataPreprocessor(args.data_path, args.dest_path, slicing, is_mat=args.is_mat, labelled_dataset=labelled_dataset)
    processor.preprocess()
