import mindspore.context as context
from mindspore.common.tensor import Tensor
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import numpy as np
import argparse
import moxing as mox

from dataset import create_cifar
from model import DenseNet
from paramter import cifar_cfg as cfg


context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')


def train_cifar10(lr_m=cfg.lr_init, momentum=cfg.momentum, num_epoch=cfg.epoch, wd=cfg.weight_decay):
    # Fetch data
    ds_train = create_cifar(num_epoch=num_epoch)
    ds_eval = create_cifar(training=False)

    # define network
    net = DenseNet(grow_rate=cfg.growth_rate)
    # define learning rate
    lr = Tensor(lr_steps(0, lr_max=lr_m, total_epochs=num_epoch, steps_per_epoch=781))
    # define loss
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    # SGD Optimizer
    opt = nn.SGD(net.trainable_params(), learning_rate=lr, momentum=momentum, weight_decay=wd)

    # model file setting
    config_ck = CheckpointConfig(save_checkpoint_steps=781 * 5, keep_checkpoint_max=60)
    ckpoint_cb = ModelCheckpoint(prefix="train_densenet_cifar10", directory="./out", config=config_ck)
    loss_cb = LossMonitor(per_print_times=1)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    # training...
    model.train(num_epoch, ds_train, callbacks=[ckpoint_cb, loss_cb])
    # testing...
    metrics = model.eval(ds_eval)
    print('Metrics:', metrics)


#  设置动态学习率，达到一半的epoch时，学习率降低为原来的十分之一，达到75%时，再次降低学习率为当前的十分之一
def lr_steps(global_step, lr_max=None, total_epochs=None, steps_per_epoch=None):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.5 * total_steps, 0.75 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr_each_step.append(lr_max)
        elif i < decay_epoch_index[1]:
            lr_each_step.append(lr_max * 0.1)
        else:
            lr_each_step.append(lr_max * 0.01)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    args, unknown = parser.parse_known_args()
    #  复制obs桶中的数据到磁盘
    mox.file.copy_parallel(src_url=args.data_url, dst_url="cifar10/")
    train_cifar10()
    #  复制模型数据到obs桶
    mox.file.copy_parallel(src_url="out/", dst_url=args.train_url)