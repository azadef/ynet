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

import argparse
from os import path, makedirs

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torchinfo import summary

from data import get_data
from networks import UNet, get_model
from utils import plot_mult, per_class_dice, mIOU


def argument_parser():
    parser = argparse.ArgumentParser()

    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--n_classes', default=9, type=int)

    # Dataset options
    parser.add_argument('--dataset', default='Duke', choices=["Duke", "UMN"])
    parser.add_argument('--image_size', default='224', type=int)

    parser.add_argument('--image_dir', default="./DukeData/")

    # Network options
    parser.add_argument('--g_ratio', default=0.5, type=float)

    # Other options
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--print_params', default=False)
    parser.add_argument('--pretrained_path', default="./pretrained_models/")

    return parser


def plot_samples_2mod(model1, model2, testset, idx_=None):
    model1.eval()
    model2.eval()
    plt.axis('off')
    plt.rcParams["text.usetex"] = True

    if idx_ is None:
        idx_ = np.random.randint(0, len(testset))

    img, label = testset.__getitem__(idx_)

    img = img.unsqueeze(0).to(device='cuda')
    label_e1 = label.unsqueeze(0).to(device='cuda')

    pred1 = model1(img)
    pred2 = model2(img)
    _, idx1 = torch.max(pred1, 1)
    _, idx2 = torch.max(pred2, 1)

    im_out = img[0][0].cpu().numpy()
    lb_np_e1 = label_e1[0][0].cpu().numpy()
    pred1_np = idx1[0].detach().cpu().numpy()
    pred2_np = idx2[0].detach().cpu().numpy()

    labels = [im_out, lb_np_e1, pred1_np, pred2_np]
    names = ["Input Image", "Expert 1", "U-Net", r'$\Upsilon$'"-Net (Ours)"]

    plot_mult(labels, names, True, idx_)


def qual_eval(testset, model, model_2):
    for i in range(len(testset)):
        plot_samples_2mod(model, model_2, testset, idx_=i)


def quant_eval(model, test_loader, n_classes, device="cuda"):
    dice = 0
    dice_all = np.zeros(n_classes)
    iou_all = 0
    counter = 0

    for img, label in tqdm.tqdm(test_loader):
        img = img.to(device=device)
        label = label.to(device=device)
        label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes)

        pred = model(img)
        max_val, idx = torch.max(pred, 1)
        pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)

        d1, d2 = per_class_dice(pred_oh, label_oh, n_classes)
        iou = mIOU(label, pred, n_classes)
        iou_all += iou
        dice += d1
        dice_all += d2

        counter += 1

    dice_all = dice_all / counter
    iou_all = iou_all / counter
    dice_all = [round(x, 2) for x in dice_all]
    dice = np.mean(dice_all[1:])
    print(" Mean Dice: ", dice, "Dice All:", dice_all, "mIoU All: ", iou_all)


def eval_unet_vs_ynet(testloader, testset, args):
    n_classes = args.n_classes

    unet_model = UNet(1, n_classes).to(args.device)
    unet_path = path.join(args.pretrained_path, "unet.pt")
    unet_model.load_state_dict(torch.load(unet_path))

    ynet_model = get_model("y_net_gen_ffc", ratio=args.g_ratio, num_classes=n_classes).to(args.device)
    ynet_path = path.join(args.pretrained_path, "y_net_gen_ffc.pt")
    ynet_model.load_state_dict(torch.load(ynet_path))

    unet_model.eval()
    ynet_model.eval()

    print("UNet Dice Score:")
    quant_eval(unet_model, testloader, n_classes=n_classes)
    print("YNet Dice Score:")
    quant_eval(ynet_model, testloader, n_classes=n_classes)
    print("Generating Qualitative Results")
    if not path.exists("./figs"):
        makedirs("./figs")
    qual_eval(testset, unet_model, ynet_model)


def print_params(n_classes):
    input_shape = (1, 1, 224, 224)

    unet_model = UNet(1, n_classes).cuda()
    ynet_model = get_model("y_net_gen", ratio=0.5).cuda()

    print("UNet")
    summary(unet_model, input_shape)

    print("YNet")
    summary(ynet_model, input_shape)


if __name__ == "__main__":
    args = argument_parser().parse_args()
    device = args.device
    data_path = args.image_dir
    img_size = args.image_size
    batch_size = args.batch_size

    if args.print_params:
        print_params(args.n_classes)

    _, _, testloader, _, _, testset = get_data(data_path, img_size, batch_size)
    eval_unet_vs_ynet(testloader, testset, args)
