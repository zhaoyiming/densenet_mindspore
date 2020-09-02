model.py : DenseNet网络模型文件(k=12, Depth=40)

model_bc.py : DenseNet-BC网络模型文件(k=12, Depth=100)

parameter.py : 包含要使用到的超参数和参数

dataset.py : 加载Cifar10数据，需要确定数据集位置

train.py : 训练文件

eval.py :  推理文件，需要确定模型位置

best_model.ckpt ：得到的最优DenseNet模型

best_model_bc.ckpt ：得到的最优DenseNet-BC模型

cifar10 : 包含模型best_model.ckpt和cifar10数据集

如要使用densenet-bc模型需要将train.py和eval.py文件中的from model 改为from model_bc，同时如要验证densenet-bc模型，请修改eval.py中的模型地址。并且需要将dataset.py中的C.Normalize方法中的参数改为[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]。