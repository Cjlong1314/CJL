# todo 代码调整过
from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np

np.random.seed(123)

class RGCNLinkDataset(object):
    def __init__(self, name, dir=None):
        self.num_nodes = None
        self.num_rels = None
        self.train = None
        self.valid = None
        self.test = None
        self.entity_dict = None
        self.relation_dict = None
        self.name = name
        # todo 拼接 路径和数据集
        if dir:
            self.dir = dir
            self.dir = os.path.join(self.dir, self.name)
        print(self.dir)

    # todo 加载数据============
    def load(self, load_time=True):
        entity_path = os.path.join(self.dir, 'entity2id.txt')
        relation_path = os.path.join(self.dir, 'relation2id.txt')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        # todo _read_dictionary将路径下的文件中的数据转换为字典数据
        entity_dict = _read_dictionary(entity_path)  # 字典类型数据，键为数字化的实体，值为实体
        relation_dict = _read_dictionary(relation_path)  # 字典类型数据，键为数字化的关系，值为关系
        # todo _read_triplets_as_list将四元组数据作为列表元素为存储到列表中
        self.train = np.array(_read_triplets_as_list(train_path, load_time))  # 训练数据<四元组>
        self.valid = np.array(_read_triplets_as_list(valid_path, load_time))  # 交叉验证数据<四元组>
        self.test = np.array(_read_triplets_as_list(test_path, load_time))  # 测试数据<四元组>
        self.num_nodes = len(entity_dict)  # 实体数量
        print("# Sanity Check:  entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)  # 关系数量
        print("# Sanity Check:  relations: {}".format(self.num_rels))
        self.entity_dict = entity_dict  # 字典类型数据，键为数字化的实体，值为实体
        self.relation_dict = relation_dict  # 字典类型数据，键为数字化的关系，值为关系
        print("# Sanity Check:  edges: {}".format(len(self.train)))


# todo 从本地根据“目录和数据集”加载数据
def load_from_local(dir, dataset):
    data = RGCNLinkDataset(dataset, dir)
    data.load()
    return data

# todo 将文件中的数据转换为字典数据
def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[int(line[1])] = line[0]
    return d


# todo 将数据根据分隔符分割，得到四元组数据
def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


# todo 将四元组数据作为列表元素为存储到列表中
def _read_triplets_as_list(filename, load_time):
    l = []
    for triplet in _read_triplets(filename):
        s = int(triplet[0])  # 头实体
        r = int(triplet[1])  # 关系
        o = int(triplet[2])  # 尾实体
        if load_time:
            st = int(triplet[3])
            l.append([s, r, o, st])  # 添加四元组
        else:
            l.append([s, r, o])  # 三元组
    return l
