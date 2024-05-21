import os
import numpy as np
import torch
import scipy.sparse as sp

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])  # 实体，关系，时间点的个数

def load_quadruples(inPath, fileName, num_r):
    quadrupleList = []
    times = set()
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([tail, rel + num_r, head, time])
    times = list(times)
    times.sort()
    return np.asarray(quadrupleList), np.asarray(times)

def load_list(inPath, entityDictPath, relationDictPath):
    entity_list = []
    relation_list = []
    with open(os.path.join(inPath, entityDictPath), 'r') as fr:
        for line in fr:
            line_split = line.split()
            # id = int(line_split[-1])
            text = line_split[0]
            if len(line_split) > 2:
                for i in line_split[1:-1]:
                    text += " " + i
            entity_list.append(text)
    with open(os.path.join(inPath, relationDictPath), 'r') as fr:
        for line in fr:
            line_split = line.split()

            # id = int(line_split[-1])

            text = line_split[0]
            if len(line_split) > 2:
                for i in line_split[1:-1]:
                    text += " " + i
            relation_list.append(text)

    return entity_list, relation_list

def get_outputs(dataset, s_list, p_list, t_list, num_rels, k, is_multi_step=False):
    """
    :param dataset: 数据集
    :param s_list: 头实体列表
    :param p_list: 关系列表
    :param t_list: 时间点列表
    :param num_rels: 关系数量
    :param k: 缩放因子
    :param is_multi_step: 是否是多步骤处理
    :return: 计算给定数据集的输出矩阵，用于模型预测和评估
    """
    outputs = []
    if not is_multi_step:
        freq_graph = sp.load_npz('./data/{}/history_seq/h_r_history_train_valid.npz'.format(dataset))
        for idx in range(len(s_list)):
            s = s_list[idx]
            p = p_list[idx]
            num_r_2 = num_rels * 2
            row = s * num_r_2 + p
            outputs.append(freq_graph[row].toarray()[0] * k)
    else:
        unique_t_list = list(set(t_list))
        tim_seq_dict = {}
        for tim in unique_t_list:
            tim_seq_dict[str(tim)] = sp.load_npz(
                './data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, tim))
        for idx in range(len(s_list)):
            s = s_list[idx]
            p = p_list[idx]
            t = t_list[idx]
            num_r_2 = num_rels * 2
            row = s * num_r_2 + p
            outputs.append(tim_seq_dict[str(t)][row].toarray()[0] * k)

    return torch.tensor(outputs)

def sort_and_rank(score, target):
    """
    :param score: 分数张量
    :param target: 目标张量
    :return: 对预测结果进行排序，并确定目标在排序结果中的位置
    """
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))  # target.view(-1, 1) 将目标实体 target 重新形状为一个二维张量（N, 1），然后进行比较返回非零元素的索引
    indices = indices[:, 1].view(-1)
    return indices

# return MRR (raw), and Hits @ (1, 3, 10)
def calc_raw_mrr(score, labels, hits=[]):
    """
    :param score: 预测的评分矩阵，形状为（N,M） N是样本数量，M是候选项的数量
    :param labels: 目标实体的索引，形状为（N,）
    :param hits: Hits@k的阈值列表
    :return: mrr：平均倒数排名 (Mean Reciprocal Rank)。
            hits1：Hits @ 1，即目标项在前 1 名中的比例。
            hits3：Hits @ 3，即目标项在前 3 名中的比例
            hits10：Hits @ 10，即目标项在前 10 名中的比例
    MRR（Mean Reciprocal Rank，平均倒数排名）是一种用于评估信息检索系统和推荐系统的指标。它衡量的是目标项在预测排序中的排名的倒数的平均值。其计算方式如下：
 。
1. 计算每个目标项的排名倒数：
如果目标项在第 𝑘个位置上，那么其倒数排名是 1/𝑘
2. 计算所有目标项的倒数排名的平均值：
对所有目标项的倒数排名取平均值，得到 MRR。
MRR 的值范围是 (0, 1]，越接近 1 表示预测越准确。
    """
    with torch.no_grad():

        ranks = sort_and_rank(score, labels)  # 计算每个样本的排名，却对目标样本在排序结果中的位置

        ranks += 1 # change to 1-indexed  # 将排名加1，使其从1开始

        mrr = torch.mean(1.0 / ranks.float())  # 计算mrr

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

#######################################################################
#
# Utility functions for evaluations (filtered)
#
#######################################################################

# 这两个函数用于在评估知识图谱补全模型时，对候选的头实体（h）或尾实体（t）进行过滤。过滤的目的是排除训练集、验证集和测试集中已经存在的三元组，
# 以便更准确地评估模型在未见过的三元组上的表现。
def filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    filtered_h = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t))
    # Do not consider an object if it is part of a triplet to filter
    for h in range(num_entities):
        if (h, target_r, target_t) not in triplets_to_filter:
            filtered_h.append(h)
    return torch.LongTensor(filtered_h)

def filter_t(triplets_to_filter, target_h, target_r, target_t, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    filtered_t = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t))
    # Do not consider an object if it is part of a triplet to filter
    for t in range(num_entities):
        if (target_h, target_r, t) not in triplets_to_filter:
            filtered_t.append(t)
    return torch.LongTensor(filtered_t)

# 这个函数 get_filtered_rank 用于计算在过滤后的候选实体集合中，目标实体的排名。这个过程是知识图谱评估的一部分，用于评估模型预测的准确性。
def get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter, entity):
    """ Perturb object in the triplets
    entity：字符串，值为 'object' 或 'subject'，表示要计算的是尾实体（object）还是头实体（subject）的排名。
    这个函数的目的是在评估模型时，计算目标实体在过滤后的候选实体集合中的排名。通过这种方式，可以更加准确地评估模型在知识图谱补全任务中的表现，
    因为它排除了那些已经在训练集、验证集和测试集中存在的三元组的影响。
    """
    num_entities = num_entity
    ranks = []

    for idx in range(test_size):
        target_h = h[idx]
        target_r = r[idx]
        target_t = t[idx]
        # print('t',target_t)
        if entity == 'object':
            filtered_t = filter_t(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_t_idx = int((filtered_t == target_t).nonzero())
            _, indices = torch.sort(score[idx][filtered_t], descending=True)
            rank = int((indices == target_t_idx).nonzero())
        if entity == 'subject':
            filtered_h = filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_h_idx = int((filtered_h == target_h).nonzero())
            _, indices = torch.sort(score[idx][filtered_h], descending=True)
            rank = int((indices == target_h_idx).nonzero())

        ranks.append(rank)
    return torch.LongTensor(ranks)


def calc_filtered_test_mrr(num_entity, score, train_triplets, valid_triplets, valid_triplets2, test_triplets, entity, hits=[]):
    """
    :param num_entity: 实体的总数。
    :param score: 模型输出的得分矩阵，形状为 (test_size, num_entities)，表示每个测试样本的每个实体的得分。
    :param train_triplets:
    :param valid_triplets:
    :param valid_triplets2: 第二个验证集中的三元组（如果有）。
    :param test_triplets:
    :param entity: 字符串，值为 'object' 或 'subject'，表示要计算的是尾实体（object）还是头实体（subject）的排名。
    :param hits: 一个包含三个整数的列表，分别表示 Hits@1, Hits@3 和 Hits@10 的计算范围。
    :return:
    函数计算过滤后的测试集上的 MRR（平均倒数排名）和 Hits@K 指标。这个函数用于评估知识图谱嵌入模型在过滤条件下的预测效果。所谓过滤条件，
    即排除训练集、验证集和测试集中已经存在的三元组的影响，只考虑模型在这些数据之外的预测能力。
    """
    with torch.no_grad():
        # 提取测试集中的头实体、关系和尾实体
        h = test_triplets[:, 0]
        r = test_triplets[:, 1]
        t = test_triplets[:, 2]
        test_size = test_triplets.shape[0]
        # 将训练集、验证集和测试集的三元组转换为张量格式
        train_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in train_triplets])
        valid_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets])
        valid_triplets2 = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets2])
        test_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in test_triplets])
        # 合并所有三元组，并转换为集合形式，以便进行过滤
        triplets_to_filter = torch.cat([train_triplets, valid_triplets, valid_triplets2, test_triplets]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        # 计算过滤后的排名
        ranks = get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter, entity)
        # 将排名从0索引转换为1索引
        ranks += 1 # change to 1-indexed
        # 计算 MRR（平均倒数排名）
        mrr = torch.mean(1.0 / ranks.float())
        # 计算 Hits@1, Hits@3 和 Hits@10
        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

#######################################################################
#
# Utility functions for evaluations (time-aware-filtered) 用于评估的实用函数（时间感知过滤）
#
#######################################################################

def ta_filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    filtered_h = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t))
    # Do not consider an object if it is part of a triplet to filter
    for h in range(num_entities):
        if (h, target_r, target_t) not in triplets_to_filter:
            filtered_h.append(h)
    return torch.LongTensor(filtered_h)

def ta_filter_t(triplets_to_filter, target_h, target_r, target_t, target_tim, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    target_tim = int(target_tim)
    filtered_t = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t, target_tim) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t, target_tim))
    # Do not consider an object if it is part of a triplet to filter
    for t in range(num_entities):
        if (target_h, target_r, t, target_tim) not in triplets_to_filter:
            filtered_t.append(t)
    return torch.LongTensor(filtered_t)

def ta_get_filtered_rank(num_entity, score, h, r, t, tim, test_size, triplets_to_filter, entity):
    """ Perturb object in the triplets
    """
    num_entities = num_entity
    ranks = []

    for idx in range(test_size):
        target_h = h[idx]
        target_r = r[idx]
        target_t = t[idx]
        target_tim = tim[idx]
        # print('t',target_t)
        if entity == 'object':
            filtered_t = ta_filter_t(triplets_to_filter, target_h, target_r, target_t, target_tim, num_entities)
            target_t_idx = int((filtered_t == target_t).nonzero())
            _, indices = torch.sort(score[idx][filtered_t], descending=True)
            rank = int((indices == target_t_idx).nonzero())
        if entity == 'subject':
            filtered_h = ta_filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_h_idx = int((filtered_h == target_h).nonzero())
            _, indices = torch.sort(score[idx][filtered_h], descending=True)
            rank = int((indices == target_h_idx).nonzero())

        ranks.append(rank)
    return torch.LongTensor(ranks)


def ta_calc_filtered_test_mrr(num_entity, score, valid_triplets2, test_triplets, entity, hits=[]):
    with torch.no_grad():
        h = test_triplets[:, 0]
        r = test_triplets[:, 1]
        t = test_triplets[:, 2]
        tim = test_triplets[:, 3]
        test_size = test_triplets.shape[0]

        valid_triplets2 = torch.Tensor([[quad[0], quad[1], quad[2], quad[3]] for quad in valid_triplets2]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in valid_triplets2}

        ranks = ta_get_filtered_rank(num_entity, score, h, r, t, tim, test_size, triplets_to_filter, entity)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

