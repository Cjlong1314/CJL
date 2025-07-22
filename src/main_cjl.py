import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import random

sys.path.append("..")

from rgcn import utils_cjl
from src.config import args
from rgcn.utils_cjl import build_sub_graph
from src.cjl import CJL
import torch.nn.modules.rnn


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, model_name, mode,
         global_history_len):
    # ranks_raw, mrr_raw_list = [], []
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []

    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name,
                                                                  checkpoint['epoch']))  # use best stat checkpoint
        print("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()
    # do not have inverse relation in test input
    # todo 选出最近 k 个历史子图
    local_input_list = [snap for snap in history_list[-args.history_len:]]
    # todo 全局历史子图
    global_input_list = [snap for snap in history_list[-global_history_len:]]

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        flattened = np.vstack(global_input_list)
        global_glist = [build_sub_graph(num_nodes, num_rels, flattened, use_cuda, args.gpu)]
        local_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in local_input_list]

        # todo 一个测试子图
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)
        test_triples, final_score = model.predict(local_glist, global_glist, num_rels,
                                                  test_triples_input, use_cuda)

        # mrr_snap, rank_raw = utils_cjl.get_total_rank(test_triples, final_score, all_ans_list[time_idx],
        #                                               eval_bz=1000, rel_predict=0)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils_cjl.get_total_rank(test_triples, final_score,
                                                                                    all_ans_list[time_idx],
                                                                                    eval_bz=1000,
                                                                                    rel_predict=0)
        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # reconstruct history graph list
        global_input_list.pop(0)
        global_input_list.append(test_snap)
        local_input_list.pop(0)
        local_input_list.append(test_snap)
    mrr_raw = utils_cjl.stat_ranks(ranks_raw, "raw_ent", model_name)
    mrr_filter = utils_cjl.stat_ranks(ranks_filter, "filter_ent", model_name)
    return mrr_raw
    # return mrr_raw, mrr_filter


def run_experiment(args):
    # load graph data
    print("loading graph data")
    # todo 加载数据
    data = utils_cjl.load_data(args.dataset)
    train_list, train_time_list = utils_cjl.split_by_time(data.train)
    valid_list, valid_time_list = utils_cjl.split_by_time(data.valid)
    test_list, test_time_list = utils_cjl.split_by_time(data.test)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    # todo 数据类型为 {{o -> r} -> s} 和 {{s -> r} -> o}
    all_ans_list_test = utils_cjl.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    # todo 数据类型为 {{o -> r} -> s} 和 {{s -> r} -> o}
    all_ans_list_valid = utils_cjl.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)

    model_name = "{}-history_len{}-history_rate{}-local_num_hidden_layers{}-global_num_hidden_layers{}-global_history_len{}".format(
        args.dataset, args.history_len, args.history_rate, args.local_num_hidden_layers,
        args.global_num_hidden_layers, args.global_history_len)
    model_state_file = '../models/' + model_name

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    # create stat
    model = CJL(num_nodes, num_rels, args.n_hidden, args.history_len, use_cuda, args.gpu, args.history_rate,
                args.local_num_hidden_layers, args.global_num_hidden_layers)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        test(model, train_list + valid_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list_test,
             model_state_file, "test", args.global_history_len)
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(
            model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        history_rate = args.history_rate
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            idx = [_ for _ in range(len(train_list))]
            # todo random.shuffle函数用于将一个序列中的所有元素随机排序, 将时间片乱序
            random.shuffle(idx)
            for train_sample_num in tqdm(idx):
                if train_sample_num == 0 or train_sample_num == 1: continue
                # todo 获取当前快照中的所有事件作为预测序列
                output = train_list[train_sample_num:train_sample_num + 1]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [
                    torch.from_numpy(_).long() for _ in output]
                # todo 获取最近的几个快照中的所有事件作为局部历史序列
                if train_sample_num - args.history_len < 0:
                    local_input_list = train_list[0: train_sample_num]
                else:
                    local_input_list = train_list[train_sample_num - args.history_len: train_sample_num]

                if train_sample_num - args.global_history_len < 0:
                    global_input_list = train_list[0: train_sample_num]
                else:
                    global_input_list = train_list[train_sample_num - args.global_history_len: train_sample_num]
                # todo 局部历史序列生成局部历史子图
                local_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in
                               local_input_list]
                # todo 全局历史序列生成全局历史子图
                flattened = np.vstack(global_input_list)
                global_glist = [build_sub_graph(num_nodes, num_rels, flattened, use_cuda, args.gpu)]
                # todo 全局和局部处理
                if history_rate == 1:
                    loss = model.get_loss(local_glist, output[0], use_cuda, "local")
                elif history_rate == 0:
                    loss = model.get_loss(global_glist, output[0], use_cuda, "global")
                else:
                    loss_local = model.get_loss(local_glist, output[0], use_cuda, "local")
                    loss_global = model.get_loss(global_glist, output[0], use_cuda, "global")
                    loss = history_rate * loss_local + (1 - history_rate) * loss_global
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print(
                "Epoch {:04d} | Ave Loss: {:.4f} | Best MRR {:.4f} | Model {} ".format(epoch, np.mean(losses), best_mrr,
                                                                                       model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw = test(model, train_list, valid_list, num_rels, num_nodes, use_cuda, all_ans_list_valid,
                               model_state_file, "train", args.global_history_len)
                if mrr_raw < best_mrr:
                    if epoch >= args.n_epochs:
                        break
                else:
                    best_mrr = mrr_raw
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        test(model, train_list + valid_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list_test,
             model_state_file, "test", args.global_history_len)
    return


if __name__ == '__main__':
    args = args
    run_experiment(args)
    sys.exit()
