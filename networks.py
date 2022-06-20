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

from collections import OrderedDict

import torch
import torch.nn as nn

from ffc import FFC_BN_ACT, ConcatTupleLayer


def get_model(model_name, in_channels=1, num_classes=9, ratio=0.5):
    if model_name == "unet":
        model = UNet(in_channels, num_classes)
    elif model_name == "y_net_gen":
        model = YNet_general(in_channels, num_classes, ffc=False)
    elif model_name == "y_net_gen_ffc":
        model = YNet_general(in_channels, num_classes, ffc=True, ratio_in=ratio)
    else:
        print("Model name not found")
        assert False

    return model

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class YNet_general(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_general, self).__init__()

        self.ffc = ffc
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge

        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_general._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_general._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_general._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_general._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_general._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_general._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_general._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_general._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = YNet_general._block(features * 8, features * 16, name="bottleneck")  # 8, 16

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 6) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 4) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 2, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))
        enc4_2 = self.pool4(enc4)

        if self.ffc:
            enc1_f = self.encoder1_f(x)
            enc1_l, enc1_g = enc1_f
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))

        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.cat_merge:
            a = torch.zeros_like(enc4_2)
            b = torch.zeros_like(enc4_f2)

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1)
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)

            bottleneck = torch.cat((enc4_2, enc4_f2), 1)
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1))

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2), 1)

        bottleneck = self.bottleneck(bottleneck)

        dec4 = self.upconv4(bottleneck)

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
