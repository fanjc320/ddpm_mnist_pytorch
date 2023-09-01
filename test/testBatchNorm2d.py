# encoding:utf-8
# https://blog.csdn.net/bigFatCat_Tom/article/details/91619977
# 基本原理
# 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定，BatchNorm2d()
# 函数数学原理如下：
# BatchNorm2d()
# 内部的参数如下：
# 1.
# num_features：一般输入参数为batch_size * num_features * height * width，即为其中特征的数量
# 2.
# eps：分母中添加的一个值，目的是为了计算的稳定性，默认为：1e-5
# 3.
# momentum：一个用于运行过程中均值和方差的一个估计参数（我的理解是一个稳定系数，类似于SGD中的momentum的系数）
# 4.
# affine：当设为true时，会给定可以学习的系数矩阵gamma和beta

import torch
import torch.nn as nn

# num_features - num_features from an expected input of size:batch_size*num_features*height*width
# eps:default:1e-5 (公式中为数值稳定性加到分母上的值)
# momentum:动量参数，用于running_mean and running_var计算的值，default：0.1
m = nn.BatchNorm2d(2, affine=True)  # affine参数设为True表示weight和bias将被使用
input = torch.randn(1, 2, 3, 4)
output = m(input)

print("input:",input)
print("weight:",m.weight)#tensor([1., 1.], requires_grad=True)
print("bias:",m.bias)#tensor([0., 0.], requires_grad=True)
print("output:",output)
print("output.size",output.size())

# 分析：输入是一个1*2*3*4 四维矩阵，gamma和beta为一维数组，是针对input[0][0]，input[0][1]两个3*4的二维矩阵分别进行处理的，我们不妨将input[0][0]的按照上面介绍的基本公式来运算，
# 看是否能对的上output[0][0]中的数据。首先我们将input[0][0]中的数据输出，并计算其中的均值和方差。
print("输入的第一个维度:")
print(input[0][0]) #这个数据是第一个3*4的二维数据
firstDimenMean=torch.Tensor.mean(input[0][0])
#求第一个维度的均值和方差
firstDimenVar=torch.Tensor.var(input[0][0],False)   #false表示贝塞尔校正不会被使用
print("m:",m) #BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
print('m.eps=',m.eps)
print("firstDimenMean:",firstDimenMean)
print("firstDimenVar:",firstDimenVar)

print("weight:",m.weight)#tensor([1., 1.], requires_grad=True)
print("bias:",m.bias)#tensor([0., 0.], requires_grad=True)
batchnormone=((input[0][0][0][0]-firstDimenMean)/(torch.pow(firstDimenVar,0.5)+m.eps))\
    *m.weight[0]+m.bias[0]
print("batchnormone:",batchnormone)
# 结果值等于output[0][0][0][0]。ok，代码和公式完美的对应起来了。