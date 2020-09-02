import mindspore.context as context
import numpy as np
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import moxing as mox
import argparse

from dataset import create_cifar
from model import DenseNet
from paramter import cifar_cfg as cfg

np.random.seed(1)
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')


def test_cifar10():
    ds_eval = create_cifar(training=False)
    net = DenseNet(grow_rate=cfg.growth_rate)
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    load_param = load_checkpoint("cifar10/best_model.ckpt")
    load_param_into_net(net, load_param)
    model = Model(net, loss, metrics={'acc', 'loss'})
    metrics = model.eval(ds_eval)
    print('Metrics:', metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of output.')
    args, unknown = parser.parse_known_args()
    #  复制obs桶中的数据到磁盘
    mox.file.copy_parallel(src_url=args.data_url, dst_url="cifar10/")
    test_cifar10()
