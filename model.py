import torch.nn as tnn
from torch import nn


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class VGG16(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        feature_extractor = tnn.Sequential(vgg_conv_block([3, 64],         [64, 64],        [3, 3],    [1, 1],    2, 2),
                                           vgg_conv_block([64, 128],       [128, 128],      [3, 3],    [1, 1],    2, 2),
                                           vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2),
                                           vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2),
                                           vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2))

        classifier_block = tnn.Sequential(vgg_fc_layer(7 * 7 * 512, 4096),
                                          vgg_fc_layer(4096, 4096),
                                          tnn.Linear(4096, n_classes))

        self._net = tnn.Sequential(feature_extractor, Flatten(), classifier_block)

    def forward(self, x):
        return self._net(x)

