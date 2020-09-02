from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.nn as nn
import numpy as np


#  初始化
def _weight_variable(shape):
    init_value = np.random.randn(*shape).astype(np.float32) * 0.01
    return Tensor(init_value)


#  3×3卷积层
def _conv3_spec(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=1, weight_init=weight, pad_mode='pad', has_bias=False)


#  1×1卷积层
def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, weight_init=weight, has_bias=False)


#  BatchNormLayer
def _bn(channel):
    return nn.BatchNorm2d(channel, gamma_init=1, beta_init=0)


#  DenseLayer
def _fc(in_channel, out_channel):
    return nn.Dense(in_channel, out_channel, has_bias=True, bias_init=0)


#  Transition
class Transition(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(Transition, self).__init__()
        
        self.bn1 = _bn(in_channel)
        self.conv1 = _conv1x1(in_channel, out_channel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Ascend 0.5版本不支持avgpool2d，使用Maxpool2d代替

    def construct(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.pool(out)
        return out


#  DenseBlock中的每个DenseLayer
class Dense_layer(nn.Cell):
    def __init__(self, in_channel, grow_rate):
        super(Dense_layer, self).__init__()
        channels = grow_rate * 4

        self.norm1 = _bn(in_channel)
        self.conv1 = _conv1x1(in_channel, channels)
        
        self.norm2 = _bn(channels)
        self.conv2 = _conv3_spec(channels, grow_rate)

        self.relu = nn.ReLU()
        self.concat = P.Concat(1)

    def construct(self, x):
        out = self.norm1(x)
        out = self.relu(out) 
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = self.relu(out) 
        out = self.conv2(out)

        out = self.concat((x, out))
        return out
    

class DenseNet(nn.Cell):
    """
      DenseNet architecture.

      Args:
          grow_rate: 对应论文中的growth_rate.
          blcok_config: 记录每个DenseBlock中的层数
          num_init_feature: 第一个卷积层中的out channels，一般为16或者growth_rate的两倍
          compression_rate: 对应论文中的压缩率，减少计算量
          num_classes : 训练数据集的类数.

    """
    def __init__(self, grow_rate=12, depth=100, num_init_feature=24, compression_rate=0.5,
                 num_class=10):
        super(DenseNet, self).__init__()
        block = (depth-4) // 6
        
        self.conv_first = _conv3_spec(3, num_init_feature)

        num_feature1 = num_init_feature
        self.layer1 = self.Dense_Block(block, num_feature1, grow_rate)
        num_feature1 += block * grow_rate
        self.tran1 = Transition(num_feature1, int(num_feature1 * compression_rate))

        num_feature2 = int(num_feature1 * compression_rate)
        self.layer2 = self.Dense_Block(block, num_feature2, grow_rate)
        num_feature2 += block * grow_rate
        self.tran2 = Transition(num_feature2, int(num_feature2 * compression_rate))

        num_feature3 = int(num_feature2 * compression_rate)
        self.layer3 = self.Dense_Block(block, num_feature3, grow_rate)
        num_feature3 += block * grow_rate

        self.norm1 = nn.BatchNorm2d(num_feature3)
        self.relu = nn.ReLU()

        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.mean = P.ReduceMean(keep_dims=True)  # Ascend 0.5版本不支持全局池化，使用ReduceMean代替

        self.dense = _fc(num_feature3, num_class)

    def construct(self, x):
        # first conv
        out = self.conv_first(x)
        # Dense Block1
        out = self.layer1(out)
        out = self.tran1(out)
        # Dense Block2
        out = self.layer2(out)
        out = self.tran2(out)
        # Dense Block3
        out = self.layer3(out)

        out = self.norm1(out)
        out = self.relu(out)
        # Global pooling
        out = self.mean(out, (2, 3))
        # Dense layer
        out = self.reshape(out, (self.shape(out)[0], -1))
        out = self.dense(out)

        return out

    #  根据参数构建Dense Blocks
    def Dense_Block(self, layer_num, num_input, grow_rate):
        layers = []

        for i in range(layer_num):
            res = Dense_layer(num_input + i * grow_rate, grow_rate)
            layers.append(res)

        return nn.SequentialCell(layers)