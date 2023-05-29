"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn
import config as cfg
""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, -1 , x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
        self.base_model = None
        self.distill_feature = cfg.DISTILL
        self.warp = cfg.WARP
        self.feature_store = None 
        self.enable_warp_train = False

    def get_features(self):
        return self.features 
    
    def adaptation(self, layer_id, num_class, in_feature, old_class):
        with torch.no_grad():
            old_weight = self.layers[layer_id].pred[1].conv.weight
            old_bias = self.layers[layer_id].pred[1].conv.bias
            # print(model.layers[22].pred[1])
            # print(model.layers[29].pred[1])
            # out_dims = cfg.BASE_CLASS + cfg.NEW_CLASS + 5
            self.layers[layer_id].pred[1] = CNNBlock(in_feature, (5 + num_class) * 3, bn_act=False, kernel_size=1)
            # self.layers[layer_id].pred[1].conv.weight[:(5 + old_class) * 3] = old_weight
            num_fea_old = 5 + old_class
            self.layers[layer_id].pred[1].conv.weight[:num_fea_old] = old_weight[:num_fea_old]
            self.layers[layer_id].pred[1].conv.weight[num_fea_old + (num_class - old_class): 2*num_fea_old + (num_class - old_class)] = old_weight[num_fea_old: 2* num_fea_old]
            self.layers[layer_id].pred[1].conv.weight[2* num_fea_old + 2 * (num_class - old_class): 3*num_fea_old + 2 * (num_class - old_class)] = old_weight[2* num_fea_old:]
            self.layers[layer_id].pred[1].conv.bias[:num_fea_old] = old_bias[:num_fea_old]
            self.layers[layer_id].pred[1].conv.bias[num_fea_old + (num_class - old_class): 2*num_fea_old + (num_class - old_class)] = old_bias[num_fea_old: 2* num_fea_old]
            self.layers[layer_id].pred[1].conv.bias[2* num_fea_old + 2 * (num_class - old_class): 3*num_fea_old + 2 * (num_class - old_class)] = old_bias[2* num_fea_old:]

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        self.features = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                # print(x.shape)
                # print(layer.pred[1].conv.weight.shape)
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                self.features.append(x)
                route_connections.append(x)
            
            elif isinstance(layer, ResidualBlock) and layer.num_repeats == 4:
                self.features.append(x)
            
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers




if __name__ == "__main__":
    num_classes = 19
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    # print(model)
    print(model.layers[15].pred[1].conv.weight.shape)
    print(model.layers[15].pred[1].conv.bias.shape)
    import torch.optim as optim
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    from utils import load_checkpoint
    load_checkpoint(
            cfg.BASE_CHECK_POINT, model, optimizer, cfg.LEARNING_RATE
    )

    model.adaptation(layer_id = 15, num_class = 20, in_feature = 1024, old_class = num_classes)
    model.adaptation(layer_id = 22, num_class = 20, in_feature = 512, old_class = num_classes)
    model.adaptation(layer_id = 29, num_class = 20, in_feature = 256, old_class = num_classes) 
    # layer1 = 
    # model.eval()
    # with torch.no_grad():
    #     old_weight = model.layers[15].pred[1].conv.weight
    #     old_bias = model.layers[15].pred[1].conv.bias
    #     # print(model.layers[22].pred[1])
    #     # print(model.layers[29].pred[1])
    #     # out_dims = cfg.BASE_CLASS + cfg.NEW_CLASS + 5
    #     model.layers[15].pred[1] = CNNBlock(1024, 25 * 3, bn_act=False, kernel_size=1)
    #     model.layers[15].pred[1].conv.weight[:72] = old_weight
    #     model.layers[15].pred[1].conv.bias[:72] = old_bias
    print(model.layers[15].pred[1].conv.weight.shape)
    # model.layers[22].pred[1] = CNNBlock(512, out_dims * 3,  kernel_size=1)
    # model.layers[29].pred[1] = CNNBlock(256, out_dims * 3,  kernel_size=1)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    # assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    # assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    # assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
