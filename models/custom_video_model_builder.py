'''MobilenetV2 in PyTorch.
See the paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks" for more details.
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import utils.weight_init_helper as init_helper
from torch.autograd import Variable
from .build import MODEL_REGISTRY


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride,
                  padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, hidden_dim):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        # hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        # if expand_ratio == 1:
        #     self.conv = nn.Sequential(
        #         # dw
        #         nn.Conv3d(hidden_dim, hidden_dim, 3, stride,
        #                   1, groups=hidden_dim, bias=False),
        #         nn.BatchNorm3d(hidden_dim),
        #         nn.ReLU6(inplace=True),
        #         # pw-linear
        #         nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
        #         nn.BatchNorm3d(oup),
        #     )
        # else:
        self.conv = nn.Sequential(
            # pw
            nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride,
                      1, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm3d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ResStage(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        num_blocks,
        hidden_dim,
    ):
        super(ResStage, self).__init__()
        self.num_blocks = num_blocks

        for i in range(self.num_blocks):
            res_block = InvertedResidual(
                dim_in if i == 0 else dim_out,
                dim_out,
                stride if i == 0 else (1, 1, 1),
                hidden_dim
            )
            self.add_module("res{}".format(i), res_block)

    def forward(self, x):
        for i in range(self.num_blocks):
            m = getattr(self, "res{}".format(i))
            x = m(x)

        return x


@MODEL_REGISTRY.register()
class MyModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(MyModel, self).__init__()
        self.conv1 = conv_bn(3, 32, (1, 1, 1))
        self.res_stage2 = ResStage(
            dim_in=32,
            dim_out=32,
            stride=(1, 2, 2),
            num_blocks=5,
            hidden_dim=72
        )
        self.res_stage3 = ResStage(
            dim_in=32,
            dim_out=72,
            stride=(1, 2, 2),
            num_blocks=10,
            hidden_dim=162
        )
        self.res_stage4 = ResStage(
            dim_in=72,
            dim_out=136,
            stride=(1, 2, 2),
            num_blocks=25,
            hidden_dim=306
        )
        self.res_stage5 = ResStage(
            dim_in=136,
            dim_out=280,
            stride=(1, 2, 2),
            num_blocks=15,
            hidden_dim=630
        )

        self.conv5 = conv_1x1x1_bn(280, 630)
        self.pool5 = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.dropout = nn.Dropout(p=0.5)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection1 = nn.Linear(630, 2048, bias=True)
        self.projection2 = nn.Linear(2048, num_classes, bias=True)

        self.act = nn.Softmax(dim=4)

        # self.head = ResNetBasicHead(
        #     dim_in=[width_per_group * 32],
        #     num_classes=num_classes,
        #     pool_size=[None],
        #     dropout_rate=0.5,
        #     act_func='softmax',  # softmax for kinetics, sigmoid for ada
        # )

        init_helper.init_weights(
            self, 0.01, True
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_stage2(x)
        x = self.res_stage3(x)
        x = self.res_stage4(x)
        x = self.res_stage5(x)
        x = self.conv5(x)
        x = self.pool5(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        x = self.dropout(x)
        x = self.projection1(x)
        x = self.projection2(x)

        x = x.view(x.shape[0], -1)
        return x


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MyModel(**kwargs)
    return model


if __name__ == "__main__":
    model = get_model(num_classes=600)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(1, 3, 16, 112, 112))
    output = model(input_var)
    print(output.shape)

# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes=1000, sample_size=224, width_mult=1.):
#         super(MobileNetV2, self).__init__()
#         block = InvertedResidual
#         input_channel = 32
#         last_channel = 1280
#         interverted_residual_setting = [
#             # t, c, n, s
#             [1,  16, 1, (1, 1, 1)],
#             [6,  24, 2, (2, 2, 2)],
#             [6,  32, 3, (2, 2, 2)],
#             [6,  64, 4, (2, 2, 2)],
#             [6,  96, 3, (1, 1, 1)],
#             [6, 160, 3, (2, 2, 2)],
#             [6, 320, 1, (1, 1, 1)],
#         ]

#         # building first layer
#         assert sample_size % 16 == 0.
#         input_channel = int(input_channel * width_mult)
#         self.last_channel = int(
#             last_channel * width_mult) if width_mult > 1.0 else last_channel
#         self.features = [conv_bn(3, input_channel, (1, 2, 2))]
#         # building inverted residual blocks
#         for t, c, n, s in interverted_residual_setting:
#             output_channel = int(c * width_mult)
#             for i in range(n):
#                 stride = s if i == 0 else (1, 1, 1)
#                 self.features.append(
#                     block(input_channel, output_channel, stride, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*self.features)

#         # building classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channel, num_classes),
#         )

#         self._initialize_weights()

#     def forward(self, x):
#         x = self.features(x)
#         x = F.avg_pool3d(x, x.data.size()[-3:])
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * \
#                     m.kernel_size[2] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()


# def get_fine_tuning_parameters(model, ft_portion):
#     if ft_portion == "complete":
#         return model.parameters()

#     elif ft_portion == "last_layer":
#         ft_module_names = []
#         ft_module_names.append('classifier')

#         parameters = []
#         for k, v in model.named_parameters():
#             for ft_module in ft_module_names:
#                 if ft_module in k:
#                     parameters.append({'params': v})
#                     break
#             else:
#                 parameters.append({'params': v, 'lr': 0.0})
#         return parameters

#     else:
#         raise ValueError(
#             "Unsupported ft_portion: 'complete' or 'last_layer' expected")
