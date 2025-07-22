import os

import numpy as np
import torch
import dgl
import rgcn.knowledge_graph as knwlgrh
from collections import defaultdict


# todo 获取真值在预测结果中的最终排名
def sort_and_rank(score, target):
    # todo 对预测结果(score)进行排序
    _, indices = torch.sort(score, dim=1, descending=True)
    # todo 获取真值在预测结果中的索引
    indices = torch.nonzero(indices == target.view(-1, 1))
    # todo 根据真值在预测结果中的索引获取其排名
    indices = indices[:, 1].view(-1)
    return indices


def r2e(triplets, num_rels):
    src, rel, dst, t = triplets.transpose()
    # get all relations
    # todo 去除关系的重复值，并拼接逆关系
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r + num_rels))
    # generate r2e
    r_to_e = defaultdict(set)  # todo 新建一个以set作为值的类型的字典，set集合元素无序且不重复
    for j, (src, rel, dst, t) in enumerate(triplets):
        # todo 关系作为键，关系对应的实体作为值
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        # todo 逆关系作为键，逆关系对应的实体作为值
        r_to_e[rel + num_rels].add(src)
        r_to_e[rel + num_rels].add(dst)
    r_len = []  # todo 其中的元素为（int, int）类型
    e_idx = []  # todo 其中的元素为list类型
    idx = 0
    for r in uniq_r:
        # todo 记录当前关系的实体索引范围（即该关系所涉及的实体的数量）
        r_len.append((idx, idx + len(r_to_e[r])))
        # todo e_idx记录当前关系的所有实体索引
        e_idx.extend(list(r_to_e[r]))
        # todo 根据当前关系涉及的实体数量更新idx
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    # todo 计算节点的入度的归一化系数
    def comp_deg_norm(g):
        # todo 获取所有节点的入度
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        # todo 默认每个节点都有一个自环边，所以入度都应 +1. 即 (A, 是, A), (B, 是, B)
        in_deg = in_deg + 1
        # todo 计算归一化系数，即每个节点入度的倒数
        norm = 1.0 / in_deg
        return norm
    src, rel, dst, t = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))  # todo src: (头实体， 尾实体)；dst: (尾实体， 头实体)
    rel = np.concatenate((rel, rel + num_rels))  # todo (关系，逆关系)
    # t = np.concatenate((t, t))
    t = np.concatenate(((t + 1), (t + 1)))

    angle = 30
    # todo 考虑周期性
    cos_t = np.cos(t + angle)
    countdown_t = 1 / t

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)  # todo 添加边，原三元组和逆三元组都加入到了其中
    norm = comp_deg_norm(g)
    # todo 初始化节点特征，分别存储节点id和对应的入度的归一化系数
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    # todo 计算边特征，为每条边计算一个权重 norm，定义为 目标节点（dst）的归一化系数 × 源节点（src）的归一化系数。
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    # todo 添加边的类型，对应知识图谱中的不同关系
    g.edata['type'] = torch.LongTensor(rel)
    g.edata['time'] = torch.LongTensor(cos_t + countdown_t)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r  # todo 关系+逆关系的种类
    g.r_len = r_len  # todo 关系+逆关系对应实体的范围
    g.r_to_e = r_to_e  # todo 关系+逆关系对应实体的索引
    if use_cuda:
        g = g.to(gpu)
        g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return g


# def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict=0):
#     num_triples = len(test_triples)
#     n_batch = (num_triples + eval_bz - 1) // eval_bz
#     rank = []
#     for idx in range(n_batch):
#         batch_start = idx * eval_bz
#         batch_end = min(num_triples, (idx + 1) * eval_bz)
#         score_batch = score[batch_start:batch_end, :]  # todo 2 * 事件数 x 实体数
#         if rel_predict == 1:
#             target = test_triples[batch_start:batch_end, 1]  # todo 获取关系真值
#         elif rel_predict == 2:
#             target = test_triples[batch_start:batch_end, 0]  # todo 获取头实体真值
#         else:
#             target = test_triples[batch_start:batch_end, 2]  # todo 获取尾实体真值
#         rank.append(sort_and_rank(score_batch, target))

#     rank = torch.cat(rank)
#     # todo 因为排名是从0开始的，所以 +1，即从第一名开始
#     rank += 1  # change to 1-indexed
#     # todo 计算排名的倒数，即 MRR
#     mrr = torch.mean(1.0 / rank.float())
#     return mrr.item(), rank

def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        # todo 获取与(s, p)相关的所有实体
        ans = list(all_ans[h.item()][r.item()])
        # todo 从ans中删除测试三元组对应的真值
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        # todo 错误答案的得分设为-10000000
        score[_][ans] = -10000000  #
    return score

def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict=0):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]  # todo 2 * 事件数 x 3
        score_batch = score[batch_start:batch_end, :]  # todo 2 * 事件数 x 实体数
        if rel_predict == 1:
            target = test_triples[batch_start:batch_end, 1]  # todo 获取关系真值
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]  # todo 获取头实体真值
        else:
            target = test_triples[batch_start:batch_end, 2]  # todo 获取尾实体真值
        rank.append(sort_and_rank(score_batch, target))
        filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    # todo 因为排名是从0开始的，所以 +1，即从第一名开始
    rank += 1  # change to 1-indexed
    filter_rank += 1
    # todo 计算排名的倒数，即 MRR
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def stat_ranks(rank_list, method, model_name):
    hits = [1, 3, 10]
    # todo 计算所有预测三元组的MRR
    total_rank = torch.cat(rank_list)
    mrr = torch.mean(1.0 / total_rank.float())
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    # todo 计算所有预测三元组的Hits@1, Hits@3, Hits@10
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
    # os.mknod("{}.txt".format(model_name))
    with open("{}.txt".format(model_name), mode='a') as fo:
        fo.write("MRR ({}): {:.6f}\n".format(method, mrr.item()))  # 写入内容并添加换行符
        for hit in hits:
            avg_count = torch.mean((total_rank <= hit).float())
            fo.write("Hits ({}) @ {}: {:.6f}\n".format(method, hit, avg_count.item()))
        fo.write("=====================================")
    fo.close()
    return mrr


# todo 字典类型数据-- {{o -> r} -> s} 即 o-r可能有对有多个s
def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r + num_rel in d[e2]:
        d[e2][r + num_rel] = set()
    d[e2][r + num_rel].add(e1)


# todo 字典类型数据-- {{s -> r} -> o} 即 s-r可能有对有多个o
def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    # todo 字典类型数据-- {{s -> o} -> r} / {{o -> s} -> r} 即 s-o（或 s-o ）可能有对有多个r
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap, time_list = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)
    return all_ans_list


# todo 将四元组按时间戳分组
def split_by_time(data):
    snapshot_list = []  # 存储快照列表，按时间戳分组
    snapshot = []  # 单个快照
    time_list = []
    snapshots_num = 0
    latest_t = 0
    # todo 将四元组按时间戳分组
    for i in range(len(data)):
        t = data[i][3]
        time_list.append(t)
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t:  # 同一时刻发生的三元组
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        # todo================
        snapshot.append(train[:4])
    # 加入最后一个shapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1
    return snapshot_list, list(set(time_list))


# todo 加载数据
def load_data(dataset):
    if dataset in ['ICEWS18', 'ICEWS14', "GDELT", "SMALL", "ICEWS14s", "ICEWS05-15", "YAGO", "WIKI"]:
        return knwlgrh.load_from_local("../data", dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


# todo 调用cuda
def cuda(tensor):
    # todo 如果张量在CPU上，则转移到GPU上
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor
