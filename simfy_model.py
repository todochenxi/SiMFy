import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SimfyDataset(Dataset):
    """
    prepare dataset
    """

    def __init__(self, quadrupleList):
        self.data = quadrupleList
        self.targets = self.get_targets()
        self.times = self.get_times()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        quad = self.data[index]
        target = self.targets[index]
        tim = self.times[index]
        return {
            'quad': torch.tensor(quad, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            't': torch.tensor(tim, dtype=torch.long),
            's': torch.tensor(quad[0], dtype=torch.long),
            'p': torch.tensor(quad[1], dtype=torch.long),
            'o': torch.tensor(quad[2], dtype=torch.long)
        }

    def get_targets(self):
        targets = []
        for quad in self.data:
            targets.append(quad[2])
        return targets

    def get_times(self):
        times = []
        for quad in self.data:
            times.append(quad[3])
        return times

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    时间信息嵌入到一个高维空间中，以便神经网络能够处理时间序列数据。其基本原理是将输入时间通过线性变换，然后应用 cos 函数，得到时间特征的嵌入表示。
    """

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim  # 嵌入维度
        self.w = nn.Linear(1, dim)  # 定义一个先行曾，将输入从1维变换到dim维
        self.reset_parameters()  # 初始化线性层的函数

    def reset_parameters(self, ):
        # np.linspace(0, (self.dim - 1) / math.sqrt(self.dim), self.dim, dtype=np.float32)：
        # 生成一个从 0 到 (self.dim - 1) / math.sqrt(self.dim) 的等差数列，共有 self.dim 个元素。
        # 1 / math.sqrt(self.dim) ** np.linspace(...)：对等差数列中的每个元素取 1 / sqrt(dim) 的幂次。
        # torch.from_numpy(...).reshape(self.dim, -1)：将生成的数组转换为 PyTorch 张量，并调整形状为 (self.dim, 1)。
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / math.sqrt(self.dim) ** np.linspace(0, (self.dim - 1) / math.sqrt(self.dim), self.dim,
                                                                     dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False  # 训练中不更新
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output  # (batch_size, dim)

class TimeSimfy(nn.Module):
    def __init__(self, num_e, num_rel, num_t, embedding_dim):
        super(TimeSimfy, self).__init__()

        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel

        # embedding initiation
        self.rel_embeds = nn.Parameter(torch.zeros(2 * self.num_rel, embedding_dim))
        # 对 self.rel_embeds 进行 Xavier 均匀初始化，并根据 ReLU 激活函数的增益值来设置初始化范围。
        # Xavier 初始化方法会将权重初始化到一个较小的范围，以便在正向传播过程中保持激活值的较小方差，从而减少梯度消失的可能性。
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, embedding_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))
        # 是一个线性层，用于整合实体、关系和时间的嵌入向量，并输出预测结果。
        self.similarity_pred_layer = nn.Linear(3 * embedding_dim, embedding_dim)
        self.weights_init(self.similarity_pred_layer)

        # time integrated
        self.time_encode = TimeEncode(embedding_dim)
        self.time_encode.reset_parameters()

        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


    @staticmethod
    def weights_init(m):
        #  方法用于初始化模型的参数。如果 m 是线性层，则使用 Xavier 初始化方法初始化其权重。
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, s, p, o, t):
        t_embeds = self.time_encode(t.float())
        preds_raw = self.tanh(self.similarity_pred_layer(self.dropout(torch.cat((self.entity_embeds[s],
                                                                                 self.rel_embeds[p], t_embeds), dim=1))))
        preds = F.softmax(preds_raw.mm(self.entity_embeds.transpose(0, 1)), dim=1)

        nce_loss = torch.sum(torch.gather(torch.log(preds), 1, o.view(-1, 1)))
        nce_loss /= -1. * o.shape[0]

        return nce_loss, preds

class Simfy(nn.Module):
    def __init__(self, num_e, num_rel, num_t, embedding_dim):
        super(Simfy, self).__init__()

        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel

        # embedding initiation
        self.rel_embeds = nn.Parameter(torch.zeros(2 * self.num_rel, embedding_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, embedding_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))

        self.similarity_pred_layer = nn.Linear(2 * embedding_dim, embedding_dim)
        self.weights_init(self.similarity_pred_layer)

        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, s, p, o):
        preds_raw = self.tanh(self.similarity_pred_layer(self.dropout(torch.cat((self.entity_embeds[s],
                                                                                 self.rel_embeds[p]), dim=1))))
        preds = F.softmax(preds_raw.mm(self.entity_embeds.transpose(0, 1)), dim=1)

        nce_loss = torch.sum(torch.gather(torch.log(preds), 1, o.view(-1, 1)))
        nce_loss /= -1. * o.shape[0]

        return nce_loss, preds


