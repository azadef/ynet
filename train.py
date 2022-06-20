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

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from data import get_data
from losses import CombinedLoss, FocalFrequencyLoss
from networks import get_model
from utils import per_class_dice


def argument_parser():
    parser = argparse.ArgumentParser()

    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--num_iterations', default=100, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--n_classes', default=9, type=int)
    parser.add_argument('--ffc_lambda', default=0, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # Dataset options
    parser.add_argument('--dataset', default='Duke', choices=["Duke", "UMN"])
    parser.add_argument('--image_size', default='224', type=int)

    parser.add_argument('--image_dir', default="./DukeData/")
    parser.add_argument('--model_name', default="y_net_gen_ffc", choices=["unet", "y_net_gen", "y_net_gen_ffc"])

    # Network options
    parser.add_argument('--g_ratio', default=0.5, type=float)

    # Other options
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--seed', default=7, type=int)

    return parser


def colored_text(st):
    return '\033[91m' + st + '\033[0m'


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def eval(val_loader, criterion, model, n_classes, dice_s=True, device="cuda", im_save=False):
    model.eval()
    loss = 0
    counter = 0
    dice = 0

    dice_all = np.zeros(n_classes)

    for img, label in tqdm.tqdm(val_loader):
        img = img.to(device)
        label = label.to(device)
        label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes)

        pred = model(img)
        max_val, idx = torch.max(pred, 1)
        pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)

        if dice_s:
            d1, d2 = per_class_dice(pred_oh, label_oh, n_classes)
            dice += d1
            dice_all += d2

        loss += criterion(pred, label.squeeze(1), device=device).item()

        counter += 1

    loss = loss / counter
    dice = dice / counter
    dice_all = dice_all / counter
    print("Validation loss: ", loss, " Mean Dice: ", dice.item(), "Dice All:", dice_all)
    return dice


def train(args):
    device = args.device
    n_classes = args.n_classes
    model_name = args.model_name
    learning_rate = args.learning_rate
    ratio = args.g_ratio
    data_path = args.image_dir
    iterations = args.num_iterations
    img_size = args.image_size
    batch_size = args.batch_size

    criterion_seg = CombinedLoss()
    criterion_ffc = FocalFrequencyLoss()

    save_name = model_name + ".pt"

    max_dice = 0
    best_test_dice = 0
    best_iter = 0

    model = get_model(model_name, ratio=ratio, num_classes=n_classes).to(device)
    model.train()

    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate,
                                 weight_decay=args.weight_decay)

    train_loader, val_loader, test_loader, _, _, _ = get_data(data_path, img_size, batch_size)

    for t in range(iterations):
        for img, label in tqdm.tqdm(train_loader):
            img = img.to(device)
            label = label.to(device)
            label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes).squeeze()

            pred = model(img)
            max_val, idx = torch.max(pred, 1)
            pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
            pred_oh = pred_oh.permute(0, 3, 1, 2)
            label_oh = label_oh.permute(0, 3, 1, 2)
            loss = criterion_seg(pred, label.squeeze(1), device=device) + args.ffc_lambda * criterion_ffc(pred_oh,
                                                                                                          label_oh)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t % 2 == 0:
            print(loss.item())

        if t % 10 == 0 or t > 45:
            print("Epoch", t, "/", iterations)
            print("Validation")
            dice = eval(val_loader, criterion_seg, model, dice_s=True, n_classes=n_classes)
            print("Expert 1 - Test")
            dice_test = eval(test_loader, criterion_seg, model, n_classes=n_classes)

            if dice > max_dice:
                max_dice = dice
                best_iter = t
                best_test_dice = dice_test
                print(colored_text("Updating model, epoch: "), t)

                torch.save(model.state_dict(), save_name)
            model.train()
    print("Best iteration: ", best_iter, "Best val dice: ", max_dice, "Best test dice: ", best_test_dice)
    return model


if __name__ == "__main__":
    args = argument_parser().parse_args()
    print(args)
    set_seed(args.seed)

    train(args)
