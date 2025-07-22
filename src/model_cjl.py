import torch.nn as nn


# todo 代码调整过

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.layers = None
        self.num_nodes = num_nodes  # 实体数
        self.h_dim = h_dim  # 200
        self.out_dim = out_dim  # 200
        self.num_rels = num_rels  # 关系数 * 2
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.self_loop = self_loop
        self.use_cuda = use_cuda
        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    # todo 隐藏层
    def build_hidden_layer(self, idx):
        raise NotImplementedError
