import torch
from torch import nn
from torch.nn import functional as F


class InfoNCE(nn.Module):
    def __init__(self, temperature):
        super(InfoNCE, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean') #初始化 InfoNCE 损失函数
        self.temperature = temperature # 温度参数，用于控制对比损失的锐化程度

    def forward(self, batch1, batch2):
        batch_size = batch1.size(0) #256
        '''对每个嵌入向量进行 L2 归一化，确保它们的模长为 1'''
        batch1 = F.normalize(batch1)  # (256, 512)
        batch2 = F.normalize(batch2)  # (256, 512)
        '''计算两个批次之间的点积相似性矩阵'''
        similarity_matrix = torch.matmul(batch1, batch2.T)  # (256,256)
        '''创建一个布尔掩码，用于区分正样本和负样本。对角线元素为 True，表示正样本对。'''
        mask = torch.eye(batch_size, dtype=torch.bool)  # (256, 256)
        assert similarity_matrix.shape == mask.shape
        '''使用掩码提取正样本和负样本的相似性'''
        positives = similarity_matrix[mask].view(batch_size, -1)  # (256,1) 提取正样本的相似性。
        negatives = similarity_matrix[~mask].view(batch_size, -1)  # (256,255) 提取负样本的相似性。
        '''将正样本和负样本的相似性拼接成一个张量'''
        logits = torch.cat([positives, negatives], dim=1) #(256,256)
        # (bs, bs)
        '''创建标签，表示每个样本的正样本是其自身，其实这是可以保证的，因为SBC过程已经确保了这一点'''
        labels = torch.zeros(batch_size, dtype=torch.long).cuda()  # (256)
        '''应用温度参数并计算交叉熵损失'''
        logits = logits / self.temperature
        loss = self.ce(logits, labels)
        # ipdb.set_trace()
        return loss, logits


class InfoNCE_v2(nn.Module):
    def __init__(self, temperature, reduction="mean"):
        super(InfoNCE_v2, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction=reduction)
        self.temperature = temperature

    def forward(self, anchor, positive):
        batch_size = anchor.size(0)
        anchor = F.normalize(anchor)
        positive = F.normalize(positive)

        similarity_matrix = torch.matmul(anchor, positive.T)  # (bs, bs)

        logits = similarity_matrix / self.temperature
        labels = torch.LongTensor([i for i in range(batch_size)]).cuda()
        loss = self.ce(logits, labels)
        return loss, logits
