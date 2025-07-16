import torch
import torch.nn as nn
import torch.nn.functional as F


# Loss function of equation (4，10) of paper "Seeking the Shape of Sound: An Adaptive Framework for Learning Voice-Face Association"

# From https://github.com/KID-7391/seeking-the-shape-of-sound   models/backbones.py

'''这个函数计算了声音和人脸嵌入向量之间的显式模态对齐损失。'''
def cross_logit(x, v):
    '''1.归一化嵌入向量,并计算L2距离'''
    # x、v就是两组embedding向量，比如声音、人脸向量，同一行对应的是同一个人的
    dist = l2dist(F.normalize(x).unsqueeze(0), v.unsqueeze(1)) #使用 l2dist 函数计算 x 和 v 之间的 L2 距离矩阵
    # 距离矩阵 dist 的形状为 [batch_size, batch_size]，其中 dist[i, j] 表示 x[i] 和 v[j] 之间的距离。
    # 默认情况下，x、v都不是单位向量， F.normalize将x变为单位向量
    '''2.构造掩码矩阵,并将对角线上的值设置为 1，表示同一身份的嵌入对。'''
    one_hot = torch.zeros(dist.size()).to(x.device) #创建一个全零矩阵 one_hot，形状与 dist 相同。
    # 全0矩阵，[batch,batch]

    one_hot.scatter_(1, torch.arange(len(x)).view(-1, 1).long().to(x.device), 1)
    # 将对角线变为1  [\] 使用 scatter_ 方法将对角线元素设置为 1，表示同一人的声音和人脸嵌入对。
    '''3.提取正样本对的距离：'''
    pos = (one_hot * dist).sum(-1, keepdim=True)
    # 使用掩码矩阵 one_hot 提取同一人样本对的距离，得到 pos。
    '''4.计算负样本对的损失：'''
    logit = (1.0 - one_hot) * (dist - pos) #计算不同人样本对的距离与正样本距离的差值，得到 logit。
    # "不同人音、脸的距离"，比"同一人" 音脸之间的距离大多少
    '''5.计算最终损失：'''
    loss = torch.log(1 + torch.exp(logit).sum(-1) + 3.4) #使用 torch.log(1 + torch.exp(logit).sum(-1) + 3.4) 计算最终的损失

    return loss


def l2dist(a, b):
    # 计算两组嵌入向量之间的 L2 距离。
    dist = (a * b).sum(-1) # 在最后一个维度上求和，相当于让最后一个维度消失
    return dist
