import argparse
import pandas as pd

from trainer import Trainer
from utils import load_tokenizer
from data_loader import DataSet

def main(args):
    tokenizer = load_tokenizer(args)

    train_file = pd.read_csv(args.data_path + "mlm_train.csv")
    dev_file = pd.read_csv(args.data_path + "mlm_dev.csv")
    train_dataset = DataSet(train_file, tokenizer, args)
    dev_dataset = DataSet(dev_file, tokenizer, args)

    trainer = Trainer(args, tokenizer, train_dataset, dev_dataset)

    if args.do_train:
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    parser.add_argument("--pred_dir", default="./preds", type=str, help="Directory that saves prediction files")
    parser.add_argument("--model_name_or_path", default="beomi/KcELECTRA-base", type=str, help="Model name or path")
    parser.add_argument("--tuned_model_path", default="./model/KcELECTRA/baseline", type=str, help="Finetuned model path")
    parser.add_argument("--data_path", default="./data/", type=str, help="Test file to predict")

    parser.add_argument("--scheduler_name", default="linear", type=str, help="learning scheduler name")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--dev_num', type=int, default=0, help="cuda device number")
    parser.add_argument('--max_stop_number', type=int, default=5, help="maximum stop number for early stop")
    parser.add_argument("--train_batch_size", default=256, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=256, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=100, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=30, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eps", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Warmup proportion for linear warmup")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="dropout rate of classification layer")

    parser.add_argument('--logging_steps', type=int, default=50, help="Log and save every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_pred", action="store_true", help="Whether to run prediction on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--multi", action="store_true", help="Whether to use multi GPUs")

    args = parser.parse_args()
    main(args)
