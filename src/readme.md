echo $SHELL

eval "$(/home/jovyan/anaconda3/bin/conda shell.bash hook)"

conda activate regcn

python main_cjl.py --dataset GDELT --history-rate 1 --global-num-hidden-layers 1 --global-history-len 1 --history-len 6 --local-num-hidden-layers 2 --gpu 0


python main_cjl.py --dataset ICEWS05-15 --history-len 1 --global-history-len 4 --local-num-hidden-layers 1 --global-num-hidden-layers 5 --history-rate 0.2 --gpu 0 

python main_cjl.py --dataset ICEWS14s --history-len 2 --global-history-len 3 --local-num-hidden-layers 3 --global-num-hidden-layers 1 --history-rate 0.6  --gpu 0

python main_cjl.py --dataset ICEWS18 --history-len 2 --global-history-len 3 --local-num-hidden-layers 1 --global-num-hidden-layers 1 --history-rate 0.6  --gpu 0 --n-epochs 301


python main_cjl.py --dataset WIKI --global-num-hidden-layers 3 --global-history-len 1 --local-num-hidden-layers 1 --history-len 2 --history-rate 0.4 --gpu 1 --n-epochs 301

python main_cjl.py --dataset YAGO --global-num-hidden-layers 3 --global-history-len 1 --local-num-hidden-layers 1 --history-len 2 --history-rate 0.8 --gpu 0 --n-epochs 31