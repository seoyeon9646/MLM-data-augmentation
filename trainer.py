import os
import time
import json
import utils
import torch
import warnings

import numpy as np
import torch.nn as nn

from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, RandomSampler

warnings.filterwarnings("ignore")

class Trainer(object):
    def __init__(self, args, tokenizer, train_dataset=None, dev_dataset=None):
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset

        self.dir_id = None

        total_steps = (len(train_dataset) // args.train_batch_size) * args.num_train_epochs if train_dataset is not None else 0
        self.model, self.optimizer, self.scheduler, self.device = utils.initialize_model(args, total_steps)

    def train(self):
        if self.train_dataset is None:
            raise Exception("train_dataset doesn't exists!")

        today = datetime.today().strftime("%y%m%d_%H%M%S")
        self.dir_id = self.args.model_dir + "/" + str(today)
        if not os.path.exists(self.dir_id):
            os.makedirs(self.dir_id)

        log = open(self.dir_id + "/log.txt", "w")

        args_dict = vars(self.args)
        with open(self.dir_id + "/params.txt", "w") as f:
            f.write(json.dumps(args_dict))
    
        scaler = GradScaler()
        self.loss_fn = nn.CrossEntropyLoss()

        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler = train_sampler, batch_size = self.args.train_batch_size)
        
        valid_dataloader = None
        if self.dev_dataset is not None:
            valid_dataloader = DataLoader(self.dev_dataset, batch_size = self.args.eval_batch_size)

        best_log = None
        loss_check, stop_count = 10000, 0
        #############################################
        # Start training process
        #############################################
        for epoch_i in range(self.args.num_train_epochs):
            t0_epoch, t0_batch = time.time(), time.time()
            total_loss, batch_loss, batch_counts = 0, 0, 0
            self.model.train()

            tmp_print = f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val F1':^9} | {'Val Precision':^9} | {'Val Recall':^9} | {'Elapsed':^9}"
            tmp_print += '\n' + '-' * 100
            print(tmp_print)
            print(tmp_print, file=log)

            for step, batch in enumerate(train_dataloader):
                with autocast():
                    batch_counts += 1
                    b_input_ids, b_label = utils.mask_tokens(self.tokenizer, batch[0])
                    b_input_ids, b_label = b_input_ids.to(self.device), b_label.to(self.device)
                    b_attn_mask = batch[1].to(self.device)
                    self.model.zero_grad()

                    outputs = self.model(b_input_ids, attention_mask=b_attn_mask, labels=b_label)
                    logits = outputs[1]

                loss_mx = b_label != -100
                logits = logits[loss_mx].view(-1, self.tokenizer.vocab_size)
                labels = b_label[loss_mx].view(-1)
                loss = self.loss_fn(logits, labels)

                batch_loss += loss.item()
                total_loss += loss.item()

                # backprop을 통해 gradient를 계산한다
                scaler.scale(loss).backward()

                # parameter 업데이트
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                scaler.unscale_(self.optimizer)
                scaler.step(self.optimizer)

                self.scheduler.step() 
                scaler.update()

                ########################################################################################
                # Print the loss values and time elapsed for every 20 batches
                ########################################################################################
                if (step % self.args.logging_steps == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    time_elapsed = time.time() - t0_batch
                    tmp_print = f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9.2} | {'-':^10.6} | {'-':^10.6} | {'-':^10.6} | {time_elapsed:^9.2f}"
                    print(tmp_print)
                    print(tmp_print, file=log)
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # train의 평균 loss 계산
            avg_train_loss = total_loss / len(train_dataloader)
            print("-" * 100)
            print("-" * 100, file=log)
            if valid_dataloader:
                result = self.evaluate(valid_dataloader)
                result.update({"train_loss":round(avg_train_loss, 4)})
                print(result)
                print(result, file=log)
                if result["loss"] < loss_check:
                    # loss가 줄어들었을 때 모델 저장
                    best_log = result
                    loss_check = result["loss"]
                    stop_count = 0
                    self.save_model()
                else:
                    stop_count += 1

                if stop_count >= self.args.max_stop_number:
                    print("EARLY STOPPED")
                    print("EARLY STOPPED", file=log)
                    log.close()
                    break
            else:
                self.save_model()

            time_elapsed = time.time() - t0_epoch
            print("Time elapsed = ", round(time_elapsed, 4))
        
        print()
        print("MODEL TRAINING END")
        print(best_log)
        log.close()

    def evaluate(self, valid_dataloader):
        self.model.eval()

        val_loss = []
        
        for batch in valid_dataloader:
            b_input_ids, b_label = utils.mask_tokens(self.tokenizer, batch[0])
            b_input_ids, b_label = b_input_ids.to(self.device), b_label.to(self.device)
            b_attn_mask = batch[1].to(self.device)

            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_attn_mask, labels=b_label)

            # calculate loss on masked tokens
            logits = outputs[1]
            loss_mx = b_label != -100
            logits = logits[loss_mx].view(-1, self.tokenizer.vocab_size)
            labels = b_label[loss_mx].view(-1)
            loss = self.loss_fn(logits, labels)

            val_loss.append(loss.mean().item())

        val_loss = np.mean(val_loss)

        # f1, precision, recall 등을 추가해야함.
        results = {
            "loss" : val_loss,
        }
        return results

    def save_model(self):
        # Save model checkpoint
        if not os.path.exists(self.dir_id):
            os.makedirs(self.dir_id)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.dir_id)
        self.tokenizer.save_pretrained(self.dir_id)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.dir_id, 'training_args.bin'))