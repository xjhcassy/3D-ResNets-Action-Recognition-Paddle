import math
from functools import partial

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Conv3D, BatchNorm


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return Conv3D(num_channels=in_planes,
                  num_filters=out_planes,
                  filter_size=3,
                  stride=stride,
                  padding=1,
                  bias_attr=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return Conv3D(num_channels=in_planes,
                  num_filters=out_planes,
                  filter_size=1,
                  stride=stride,
                  bias_attr=False)


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = BatchNorm(num_channels=planes, act='relu')
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = BatchNorm(num_channels=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = fluid.layers.relu(out)

        return out


class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = BatchNorm(num_channels=planes, act='relu')
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = BatchNorm(num_channels=planes, act='relu')
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm(num_channels=planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = fluid.layers.relu(out)

        return out


class ResNet(fluid.dygraph.Layer):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = Conv3D(num_channels=n_input_channels,
                            num_filters=self.in_planes,
                            filter_size=(conv1_t_size, 7, 7),
                            stride=(conv1_t_stride, 2, 2),
                            padding=(conv1_t_size // 2, 3, 3),
                            bias_attr=False)
        self.bn1 = BatchNorm(self.in_planes, act='relu')

        # self.maxpool = Pool3D(pool_size=3,
        #                       pool_type='max',
        #                       pool_stride=2,
        #                       pool_padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.fc = Linear(block_inplanes[3] * block.expansion, n_classes, act='softmax')
        self.last_feature_size = block_inplanes[3] * block.expansion

        for m in self.sublayers():
            if isinstance(m, Conv3D):
                m.weight.initializer = fluid.initializer.MSRAInitializer(uniform=True,fan_in = None,seed = 0)
                # nn.init.kaiming_normal_(m.weight,
                #                         mode='fan_out',
                #                         nonlinearity='relu')
            elif isinstance(m, BatchNorm):
                m.weight.initializer=fluid.initializer.ConstantInitializer(value=1)
                m.bias.initializer=fluid.initializer.ConstantInitializer(value=0)
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = fluid.layers.pool3d(input=x, pool_size=1, pool_type='avg', stride=stride)
        zero_pads = fluid.layers.zeros([out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4)],dtype='float32')

        # if isinstance(out.data, torch.cuda.FloatTensor):
        #     zero_pads = zero_pads.cuda()

        out = fluid.layers.concat(input=[out.data, zero_pads], axis=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = fluid.dygraph.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    BatchNorm(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return fluid.dygraph.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if not self.no_max_pool:
            x = fluid.layers.pool3d(input=x,
                                    pool_size=3,
                                    pool_type='max',
                                    pool_stride=2,
                                    pool_padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = fluid.layers.adaptive_pool3d(input=x,
                                         pool_size=[1,1,1],
                                         pool_type='avg')

        # x = x.view(x.size(0), -1)
        x = fluid.layers.reshape(x, shape=[x.shape[0], -1])
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

