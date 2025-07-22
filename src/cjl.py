import torch
import torch.nn as nn
import torch.nn.functional as F
from src.decoder_cjl import ConvTransE
from src.model_cjl import BaseRGCN
import dgl.function as fn


class CJL(nn.Module):
    def __init__(self, num_ents, num_rels, h_dim, history_len, use_cuda=False, gpu=0, history_rate=0,
                 local_num_hidden_layers=1, global_num_hidden_layers=1):
        super(CJL, self).__init__()
        self.history_rate = history_rate
        self.history_len = history_len  # todo 历史长度
        self.self_loop = True
        self.h_dim = h_dim
        self.gpu = gpu
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.input_dropout = 0.2
        self.hidden_dropout = 0.2
        self.feat_dropout = 0.2
        self.num_bases = 100
        self.num_basis = 100
        self.local_num_hidden_layers = local_num_hidden_layers
        self.global_num_hidden_layers = global_num_hidden_layers
        # self.num_hidden_layers = num_hidden_layers
        self.dropout = 0.2
        self.layer_norm = True
        # todo 局部实体初始化嵌入
        self.local_ents_embedding = torch.nn.Parameter(torch.Tensor(num_ents, h_dim),
                                                       requires_grad=True).float()  # todo 实体数 x 200
        torch.nn.init.normal_(self.local_ents_embedding)
        # todo 全局实体初始化嵌入
        self.global_ents_embedding = torch.nn.Parameter(torch.Tensor(num_ents, h_dim),
                                                        requires_grad=True).float()  # todo 实体数 x 200
        torch.nn.init.normal_(self.global_ents_embedding)

        # todo 局部隐藏层初始化嵌入
        # self.local_hidden_layer = torch.nn.Parameter(torch.Tensor(h_dim, h_dim),
        #                                              requires_grad=True).float()  # todo 200 x 200
        # torch.nn.init.xavier_normal_(self.local_hidden_layer)
        #
        # # todo 全局隐藏层初始化嵌入
        # self.global_hidden_layer = torch.nn.Parameter(torch.Tensor(h_dim, h_dim),
        #                                               requires_grad=True).float()  # todo 200 x 200
        # torch.nn.init.xavier_normal_(self.global_hidden_layer)

        # todo 局部关系初始化嵌入
        self.local_rels_embedding = torch.nn.Parameter(torch.Tensor(2 * num_rels, h_dim),
                                                       requires_grad=True).float()  # todo 2 * 关系数 x 200
        torch.nn.init.xavier_normal_(self.local_rels_embedding)

        # todo 全局关系初始化嵌入
        self.global_rels_embedding = torch.nn.Parameter(torch.Tensor(2 * num_rels, h_dim),
                                                        requires_grad=True).float()  # todo 2 * 关系数 x 200
        torch.nn.init.xavier_normal_(self.global_rels_embedding)

        # todo 局部时间门权重和偏置
        self.local_time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))  # 200 x 200
        nn.init.xavier_uniform_(self.local_time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.local_time_gate_bias = nn.Parameter(torch.Tensor(h_dim))  # 200 x 1
        nn.init.zeros_(self.local_time_gate_bias)
        # GRU cell for relation evolving
        self.local_relation_cell = nn.GRUCell(h_dim * 2, h_dim)  # 400 x 200
        # todo 全局门权重和偏置
        self.global_relation_cell = nn.GRUCell(h_dim * 2, h_dim)  # 400 x 200

        # todo time embedding
        self.time_embedding = nn.Parameter(torch.Tensor(h_dim, h_dim))  # 200 x 200
        nn.init.xavier_uniform_(self.time_embedding, gain=nn.init.calculate_gain('relu'))

        # todo 损失函数
        self.loss_e = torch.nn.CrossEntropyLoss()
        self.decoder_ob = ConvTransE(num_ents, h_dim, self.input_dropout, self.hidden_dropout, self.feat_dropout)

        # self.rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, self.num_bases, self.num_basis,
        #                      self.num_hidden_layers, self.dropout, self.self_loop, use_cuda)
        self.local_rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, self.num_bases, self.num_basis,
                                   self.local_num_hidden_layers, self.dropout, self.self_loop, use_cuda)
        self.global_rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, self.num_bases, self.num_basis,
                                    self.global_num_hidden_layers, self.dropout, self.self_loop, use_cuda)

    def forward(self, history_glist, use_cuda, mode):
        if mode == "local":
            # todo 局部处理
            self.h = F.normalize(self.local_ents_embedding) if self.layer_norm else self.local_ents_embedding[:,
                                                                                    :]  # 实体数 x 200
            history_embs = []
            for i, g in enumerate(history_glist):
                g = g.to(self.gpu)
                # todo g.r_to_e是关系和逆关系对应的实体列表
                temp_e = self.h[g.r_to_e]  # 实体数 x 200
                x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(
                    self.num_rels * 2, self.h_dim).float()  # 2 * 关系数 x 200
                # todo g.r_len记录关系和逆关系所涉及的实体数量范围，g.uniq_r是图 g 中的所有关系
                # todo 对应公式 6 中的均池化======
                for span, r_idx in zip(g.r_len, g.uniq_r):  # todo zip()用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组
                    x = temp_e[span[0]:span[1], :]  # todo 获取与当前关系相关联的实体
                    x_mean = torch.mean(x, dim=0, keepdim=True)  # todo 对关系相关联的实体取均值，作为该关系的聚合特征
                    x_input[r_idx] = x_mean  # todo 索引r_idx是关系，值x_mean是关系涉及的实体的均值
                if i == 0:
                    # todo 对应公式 6 的拼接======
                    x_input = torch.cat((self.local_rels_embedding, x_input), dim=1)  # 2 * 关系数 x 400
                    # todo 对应公式 7======
                    self.h_0 = self.local_relation_cell(x_input, self.local_rels_embedding)  # 第1层输入(2 * 关系数 x 200)
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                else:
                    x_input = torch.cat((self.local_rels_embedding, x_input), dim=1)  # 2 * 关系数 x 400
                    # todo 对应公式 7(相邻子图之间的关系更新)======
                    self.h_0 = self.local_relation_cell(x_input, self.h_0)  # 第2层输出==下一时刻第一层输入(2 * 关系数 x 200)
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0  # 2 * 关系数 x 200
                # todo ============
                current_h = self.local_rgcn.forward(g, self.h, self.h_0, self.time_embedding)  # 实体数 x 200
                current_h = F.normalize(current_h) if self.layer_norm else current_h
                # todo 对应公式 5=======
                time_weight = F.sigmoid(
                    torch.mm(self.h, self.local_time_gate_weight) + self.local_time_gate_bias)  # 实体数 x 200
                # todo 对应公式 4=======
                self.h = time_weight * current_h + (1 - time_weight) * self.h  # 实体数 x 200
                history_embs.append(self.h)
            return history_embs, self.h_0
        else:
            # todo 全局处理
            self.global_h = F.normalize(self.global_ents_embedding) if self.layer_norm else self.global_ents_embedding[
                                                                                            :, :]  # 实体数 x 200

            history_embs = []
            for i, g in enumerate(history_glist):
                g = g.to(self.gpu)
                # todo g.r_to_e是关系和逆关系对应的实体列表
                temp_e = self.global_h[g.r_to_e]  # 实体数 x 200
                x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(
                    self.num_rels * 2, self.h_dim).float()  # 2 * 关系数 x 200
                # todo g.r_len记录关系和逆关系所涉及的实体数量范围，g.uniq_r是图 g 中的所有关系
                # todo 对应公式 6 中的均池化======
                for span, r_idx in zip(g.r_len, g.uniq_r):  # todo zip()用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组
                    x = temp_e[span[0]:span[1], :]  # todo 获取与当前关系相关联的实体
                    x_mean = torch.mean(x, dim=0, keepdim=True)  # todo 对关系相关联的实体取均值，作为该关系的聚合特征
                    x_input[r_idx] = x_mean  # todo 索引r_idx是关系，值x_mean是关系涉及的实体的均值
                # todo 对应公式 6 的拼接======
                x_input = torch.cat((self.global_rels_embedding, x_input), dim=1)  # 2 * 关系数 x 400
                # todo 对应公式 7======
                self.h_1 = self.global_relation_cell(x_input, self.global_rels_embedding)  # 第1层输入(2 * 关系数 x 200)
                self.h_1 = F.normalize(self.h_1) if self.layer_norm else self.h_1
                current_h = self.global_rgcn.forward(g, self.global_h, self.h_1, self.time_embedding)  # 实体数 x 200
                self.global_h = F.normalize(current_h) if self.layer_norm else current_h
                history_embs.append(self.global_h)
                return history_embs, self.h_1
        # todo==========================================

    def predict(self, local_history_glist, global_history_glist, num_rels, test_data, use_cuda):
        with torch.no_grad():
            # todo 反转三元组，更新逆关系，从而获得逆三元组
            inverse_test_triplets = test_data[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # todo 换成逆关系
            # todo 拼接原三元组和逆三元组
            test_data = test_data[:, [0, 1, 2]]
            all_triples = torch.cat((test_data, inverse_test_triplets))
            if self.history_rate == 0:
                global_evolve_embs, global_r_emb = self.forward(global_history_glist, use_cuda, "global")
                global_embedding = F.normalize(global_evolve_embs[-1]) if self.layer_norm else global_evolve_embs[-1]
                global_score = self.decoder_ob.forward(global_embedding, global_r_emb, all_triples,
                                                       mode="test")  # todo 2 * 事件数 x 实体数
                return all_triples, global_score
            elif self.history_rate == 1:
                local_evolve_embs, local_r_emb = self.forward(local_history_glist, use_cuda, "local")
                local_embedding = F.normalize(local_evolve_embs[-1]) if self.layer_norm else local_evolve_embs[-1]
                local_score = self.decoder_ob.forward(local_embedding, local_r_emb, all_triples,
                                                      mode="test")  # todo 2 * 事件数 x 实体数
                return all_triples, local_score
            else:
                local_evolve_embs, local_r_emb = self.forward(local_history_glist, use_cuda, "local")
                local_embedding = F.normalize(local_evolve_embs[-1]) if self.layer_norm else local_evolve_embs[-1]
                global_evolve_embs, global_r_emb = self.forward(global_history_glist, use_cuda, "global")
                global_embedding = F.normalize(global_evolve_embs[-1]) if self.layer_norm else global_evolve_embs[-1]
                local_score = self.decoder_ob.forward(local_embedding, local_r_emb, all_triples,
                                                      mode="test")  # todo 2 * 事件数 x 实体数
                global_score = self.decoder_ob.forward(global_embedding, global_r_emb, all_triples,
                                                       mode="test")  # todo 2 * 事件数 x 实体数
                score = self.history_rate * local_score + (1 - self.history_rate) * global_score
                return all_triples, score

    def get_loss(self, history_glist, triples, use_cuda, mode):
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        # todo 生成逆三元组
        inverse_triples = triples[:, [2, 1, 0]]
        triples = triples[:, [0, 1, 2]]
        # todo 重新编号逆关系
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        # todo 拼接三元组和逆三元组(直接预测头实体和尾实体)
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)
        # todo evolve_embs是个列表，其中的元素为每个子图生成的实体嵌入（实体数 x 200）; r_emb（2 * 关系数 x 200）
        evolve_embs, r_emb = self.forward(history_glist, use_cuda, mode)
        # todo pre_emb为最新子图的实体嵌入
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]  # 实体数 x 200
        scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)  # batch_size x 实体数
        # todo 对应公式 13=======
        loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
        return loss_ent


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, activation=act, dropout=self.dropout,
                              self_loop=self.self_loop)

    # todo=====================
    def forward(self, g, init_ent_emb, init_rel_emb, init_time_emb):
        node_id = g.ndata['id'].squeeze()  # todo 实体id
        g.ndata['h'] = init_ent_emb[node_id]  # todo 实体嵌入(实体数 x 200)
        # todo=================
        x, r, t = init_ent_emb, init_rel_emb, init_time_emb
        for i, layer in enumerate(self.layers):
            # todo=================
            layer.forward(g, r, t)
        return g.ndata.pop('h')


class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, activation=None, dropout=0.0, self_loop=False, ):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat  # 200
        self.out_feat = out_feat  # 200
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels  # 关系数
        self.rel_emb = None
        self.time_emb = None
        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))  # 200 x 200
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))  # 200 x 200
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))  # 200 x 200
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # todo 处理图中邻居信息的复制与聚合
    def propagate(self, g):
        # todo update_all():用于在图神经网络中处理邻居信息的复制与聚合。
        # todo msg_func():定义消息生成函数，为每条边计算消息。
        # todo fn.sum():定义消息聚合函数，将目标节点接收的所有消息求和，结果存到 h 字段。
        # todo apply_func():定义节点更新函数，进一步处理聚合后的 h。
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, emb_rel, emb_time):
        self.rel_emb = emb_rel  # 2 * 关系数 x 200
        self.time_emb = emb_time
        if self.self_loop:
            # todo 选择入度大于0的节点构成新的张量
            masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                                               (g.in_degrees(range(g.number_of_nodes())) > 0))
            # todo 所有节点分配一个 evolve_loop_weight 权重，入度大于0的节点额外分配一个 loop_weight 权重
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)  # 实体数 * 200
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]  # 实体数 * 200
        # calculate the neighbor message with weight_neighbor
        # todo 对应公式 8===========
        self.propagate(g)
        node_repr = g.ndata['h']
        if self.self_loop:
            node_repr = node_repr + loop_message  # todo 加入了自环边========还不太懂
        # todo 激活函数 F.rrelu
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        # todo edge_type：图中的边，node：图中边的头节点
        edge_type = edges.data['type']  # 图中边数
        # todo==================
        edge_time = edges.data['time']
        e_time = self.time_emb.index_select(0, edge_time).view(-1, self.out_feat)  # 图中边数 x 200
        # todo index_select(): 通过指定维度和索引来选择数据
        relation = self.rel_emb.index_select(0, edge_type).view(-1, self.out_feat)  # 图中边数 x 200
        node = edges.src['h'].view(-1, self.out_feat)  # 图中边数 x 200
        # todo=================
        # after add inverse edges, we only use message pass when h as tail entity
        # 这里计算的是每个节点发出的消息，节点发出消息时其作为头实体
        # todo 将头实体特征与关系嵌入相加，模拟类似 TransE 的结构（h + r ≈ t）
        msg = node + relation * e_time  # 图中边数 x 200
        # calculate the neighbor message with weight_neighbor
        msg = torch.mm(msg, self.weight_neighbor)  # 图中边数 x 200
        return {'msg': msg}

    def apply_func(self, nodes):
        # todo 利用范数更新节点
        return {'h': nodes.data['h'] * nodes.data['norm']}
