import torch
import deepspeed

import os
import json
import random
import warnings
import argparse
import numpy as np
import pandas as pd

from ds_trainer import Trainer
from utils import load_tokenizer
from data_loader import DataSet

warnings.filterwarnings(action="ignore")

seed_val = 1
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description = "deep speed learning")
parser.add_argument("--workers", default=4, type=int)
parser.add_argument("--world_size", default=4),
parser.add_argument("--rank", default=-1)
parser.add_argument("--gpu", default=None, type=int)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--model_name_or_path", default="beomi/KcELECTRA-base", type=str, help="Model name or path")
parser.add_argument("--data_path", default="./data/", type=str, help="Model name or path")
parser.add_argument("--model_dir", default="./model/", type=str, help="Model name or path")
parser.add_argument("--scheduler_name", default="linear", type=str, help="learning scheduler name")

parser.add_argument('--dev_num', type=int, default=0, help="cuda device number")
parser.add_argument('--max_stop_number', type=int, default=5, help="maximum stop number for early stop")
parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for evaluation.")
parser.add_argument("--max_seq_len", default=100, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--learning_rate", default=2e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs", default=100, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--eps", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Warmup proportion for linear warmup")
parser.add_argument('--logging_steps', type=int, default=50, help="Log and save every X updates steps.")

parser = deepspeed.add_config_arguments(parser)


def main_worker(gpu, ngpus_per_node, args):
    ###################################
    # GPU 있는지 여부 확인
    ##################################
    try:
        torch.multiprocessing.set_start_method('spawn')
    except:
        print(torch.multiprocessing.get_start_method())

    args.gpu = gpu
    args.rank = args.local_rank
    deepspeed.init_distributed()

    args.world_size = ngpus_per_node
    args.workers = 2

    tokenizer = load_tokenizer(args)
    train_file = pd.read_csv(args.data_path + "mlm_train.csv")
    dev_file = pd.read_csv(args.data_path + "mlm_dev.csv")
    train_dataset = DataSet(train_file, tokenizer, args)
    dev_dataset = DataSet(dev_file, tokenizer, args)
    
    trainer = Trainer(args, tokenizer, train_dataset, dev_dataset)
    trainer.train()

def main():
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.gpu = args.local_rank
    main_worker(args.gpu, ngpus_per_node, args)

if __name__ == "__main__":
    main()