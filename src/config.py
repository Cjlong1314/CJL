import argparse

parser = argparse.ArgumentParser(description='CJL')
# todo 明确调用的参数
parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
parser.add_argument("--history-len", type=int, default=10, help="history length")
parser.add_argument("--evaluate-every", type=int, default=5, help="perform evaluation every n epochs")
parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--test", action='store_true', default=False, help="load stat from dir and directly test")
parser.add_argument("--n-epochs", type=int, default=30, help="number of minimum training epochs on each time step")
parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--input-dropout", type=float, default=0.2, help="input dropout for decoder ")
parser.add_argument("--hidden-dropout", type=float, default=0.2, help="hidden dropout for decoder")
parser.add_argument("--feat-dropout", type=float, default=0.2, help="feat dropout for decoder")
parser.add_argument("--history-rate", type=float, default=0.2, help="feat dropout for decoder")
parser.add_argument("--local-num-hidden-layers", type=int, default=1, help="feat dropout for decoder")
parser.add_argument("--global-num-hidden-layers", type=int, default=1, help="feat dropout for decoder")
parser.add_argument("--global-history-len", type=int, default=1, help="feat dropout for decoder")

# todo 指令参数中有

args = parser.parse_args()
print(args)
