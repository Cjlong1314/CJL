from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
import math
import os

path_dir = os.getcwd()


# todo 用来预测实体
class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0,
                 channels=50, kernel_size=3, use_bias=True):

        super(ConvTransE, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                                     padding=int(math.floor(
                                         kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)  # 10000 x 200
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, triplets, nodes_id=None, mode="train", negative_rate=0,
                partial_embeding=None):
        e1_embedded_all = F.tanh(embedding)  # 实体数 x 200
        batch_size = len(triplets)  # todo 子图中的三元组＋逆三元组
        # todo 有别于ConvTransR=========>
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)  # 头实体嵌入(batch_size x 1 x 200)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)  # 关系嵌入(batch_size x 1 x 200)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)  # batch_size x 2 x 200
        # todo 有别于ConvTransR=========|
        stacked_inputs = self.bn0(stacked_inputs)  # batch_size x 2 x 200
        x = self.inp_drop(stacked_inputs)  # batch_size x 2 x 200
        x = self.conv1(x)  # batch_size x 50 x 200
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)  # batch_size x 10000
        x = self.fc(x)  # batch_size x 200
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        # todo 有别于ConvTransR=========>
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))  # batch_size x 实体数
        else:
            x = torch.mm(x, partial_embeding.transpose(1, 0))
        # todo 有别于ConvTransR=========|
        return x
