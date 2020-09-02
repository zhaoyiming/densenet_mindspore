import mindspore.dataset.transforms.c_transforms as C2
from paramter import cifar_cfg as cfg
import mindspore as ms
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.transforms.vision.c_transforms as C
from mindspore.common import dtype as mstype
import numpy as np

np.random.seed(1)
DATA_DIR_TRAIN = "cifar10/cifar-10-batches-bin/train"  # 训练集信息
DATA_DIR_TEST = "cifar10/cifar-10-batches-bin/test"  # 测试集信息


def create_cifar(training=True, num_epoch=100, batch_size=cfg.batch_size):
    ds = ms.dataset.Cifar10Dataset(DATA_DIR_TRAIN if training else DATA_DIR_TEST)

    buffer_size = 100
    rescale = 1.0 / 255.0
    shift = 0.0
    # 数据增强
    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4))
    random_horizontal_flip_op = C.RandomHorizontalFlip()
    
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
#     normalize_op = C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    change_swap_op = C.HWC2CHW()

    trans = []
    if training:
        trans += [random_crop_op, random_horizontal_flip_op]
    trans += [rescale_op, normalize_op, change_swap_op]
    type_cast_op = C2.TypeCast(mstype.int32)
    # define map operations
    ds = ds.map(input_columns="image", operations=trans)
    ds = ds.map(input_columns="label", operations=type_cast_op)
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True).repeat(num_epoch)

    return ds
