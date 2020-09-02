from easydict import EasyDict as edict
# 参数设置
cifar_cfg = edict({
    'growth_rate': 12,
    'num_classes': 10,
    'lr_init': 0.1,
    'batch_size': 64,
    'epoch': 300,
    'momentum': 0.9,
    'weight_decay': 1e-4
})