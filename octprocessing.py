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


import numpy as np


def get_valid_idx(manualLayer):
    """
    get idx taken from 
    'https://www.kaggle.com/gmonge/data-convert-and-preprocessing'
    """
    idx = []
    for i in range(0,61):
        temp = manualLayer[:,:,i]
        if np.sum(temp) != 0:
            idx.append(i)
    return idx


def get_unlabelled_idx(manualLayer):
    """
    adapted from 
    'https://www.kaggle.com/gmonge/data-convert-and-preprocessing'
    """
    idx = []
    for i in range(0,61):
        temp = manualLayer[:,:,i]
        if np.sum(temp) == 0:
            idx.append(i)
    return idx


def get_valid_img_seg(mat):
    """
    data preparation taken from (slightly modfified: fluid_class const added + img dims reduced)
    'https://www.kaggle.com/gmonge/data-convert-and-preprocessing'
    
    breaks for Subject_04.mat
    """    
    fluid_class = 9

    manualLayer = np.array(mat['manualLayers1'], dtype=np.uint16)
    manualFluid = np.array(mat['manualFluid1'], dtype=np.uint16)
    img = np.array(mat['images'], dtype=np.uint8)
    valid_idx = get_valid_idx(manualLayer)


    manualFluid = manualFluid[:, :, valid_idx]
    manualLayer = manualLayer[:, :, valid_idx]

    seg = np.zeros((496, 768, 11))
    seg[manualFluid > 0] = fluid_class
    
    max_col = -100
    min_col = 900
    for b_scan_idx in range(0, 11):
        for col in range(768):
            cur_col = manualLayer[:, col, b_scan_idx]
            if np.sum(cur_col) == 0:
                continue

            max_col = max(max_col, col)
            min_col = min(min_col, col)

            labels_idx = cur_col.tolist()
            last_st = None
            last_ed = None
            for label, (st, ed) in enumerate(zip([0]+labels_idx, labels_idx+[-1])):
                
                if st == 0 and ed == 0:
                    st = last_ed
                    print("val",seg[st, col, b_scan_idx])
                    while(seg[st, col, b_scan_idx] == fluid_class):
                        st += 1

                    while(seg[st, col, b_scan_idx] != fluid_class):
                        seg[st, col, b_scan_idx] = label
                        st += 1
                        if st >= 496:
                            break
                    continue
                if ed == 0:
                    ed = st + 1
                    while(seg[ed, col, b_scan_idx] != fluid_class):
                        ed += 1

                if st == 0 and label != 0:
                    st = ed-1
                    while(seg[st, col, b_scan_idx] != fluid_class):
                        st -= 1
                    st += 1

                seg[st:ed, col, b_scan_idx] = label
                last_st = st
                last_ed = ed

    seg[manualFluid > 0] = fluid_class

    seg = seg[:, min_col:max_col+1]
    img = img[:, min_col:max_col+1]

    # only return images with corresponding masks
    img = img[:,:,valid_idx]
    return img, seg


# simple control implementation as source of jupyter notebook unknown
def get_valid_img_seg_reimpl(scan_obj):
    fluid_class = 9
    
    manualLayer = np.array(scan_obj['manualLayers1'], dtype=np.uint16)
    manualFluid = np.array(scan_obj['manualFluid1'], dtype=np.uint16)
    img = np.array(scan_obj['images'], dtype=np.uint8)
    valid_idx = get_valid_idx(manualLayer)
    
    manualFluid = manualFluid[:, :, valid_idx]
    manualLayer = manualLayer[:, :, valid_idx]

    seg = np.zeros_like(manualFluid, dtype=np.uint8) 
    
    for bsc in range(seg.shape[2]):
        for asc in range(seg.shape[1]):
            class_idx = manualLayer[:,asc, bsc] # idx range of class i [class_idx[i-1], class_idx[i])
            
            # sometimes they use 0 idx mistakenly for empty classes 
            # ie instead of [..,123,123,150,..] -> class i has 0 innstances they use [..,123,0,150,..]
            for i,_ in enumerate(class_idx):
                if i > 0 and class_idx[i] < class_idx[i-1]:
                    class_idx[i] = class_idx[i-1]


            for label, (idx_prev, idx_cur) in enumerate(zip([0,*class_idx], [*class_idx, seg.shape[0]])):
                seg[idx_prev:idx_cur, asc, bsc] = label
    
    seg[manualFluid > 0] = fluid_class
    
    a_scan_used, = np.where(np.sum(manualLayer, axis=(0,2)) != 0)
    seg = seg[:, a_scan_used[0]:a_scan_used[-1] + 1]
    img = img[:, a_scan_used[0]:a_scan_used[-1] + 1]

    # only return images with corresponding masks
    img = img[:,:,valid_idx]

    return img, seg


def get_unlabelled_bscans(scan_obj):
    manualLayer = np.array(scan_obj['manualLayers1'], dtype=np.uint16)
    img = np.array(scan_obj['images'], dtype=np.uint8)

    valid_idx = get_unlabelled_idx(manualLayer)
    return img[:,:,valid_idx]
