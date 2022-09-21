import os
import time
import json
import utils
import torch
import warnings

import numpy as np
import torch.nn as nn

from datetime import datetime
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, DistributedSampler

warnings.filterwarnings("ignore")

class Trainer(object):
    def __init__(self, args, tokenizer, train_dataset=None, dev_dataset=None):
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset

        self.dir_id = None
        self.device = torch.device(args.gpu)
        self.model, self.optimizer, self.scheduler = utils.initialize_model_with_ds(args)

    def train(self):
        if self.train_dataset is None:
            raise Exception("train_dataset doesn't exists!")
        
        #########################################################
        # 1. 데이터로더 생성 - 학습 샘플러는 DistributedSampler로 사용해야함
        train_sampler = DistributedSampler(self.train_dataset, rank=self.args.rank, num_replicas=self.args.world_size, shuffle=True)
        train_dataloader = DataLoader(self.train_dataset, sampler = train_sampler, batch_size = self.args.train_batch_size)
        
        valid_dataloader = None
        if self.dev_dataset is not None:
            valid_dataloader = DataLoader(self.dev_dataset, batch_size = self.args.eval_batch_size)
        #########################################################


        #########################################################
        # 2. 모델을 저장할 폴더 생성 및 로깅, 파라미터 저장 파일 생성
        today = datetime.today().strftime("%y%m%d_%H%M%S")
        self.dir_id = self.args.model_dir + "/" + str(today)
        if not os.path.exists(self.dir_id):
            os.makedirs(self.dir_id)

        log = open(self.dir_id + "/log.txt", "w")

        args_dict = vars(self.args)
        with open(self.dir_id + "/params.txt", "w") as f:
            f.write(json.dumps(args_dict))
        #########################################################
    
        self.loss_fn = nn.CrossEntropyLoss()

        best_log = None
        loss_check, stop_count = 10000, 0
        #############################################
        # 3. 학습 프로세스 시작
        #############################################
        for epoch_i in range(self.args.num_train_epochs):
            train_sampler.set_epoch(epoch_i) 

            t0_epoch, t0_batch = time.time(), time.time()
            total_loss, batch_loss, batch_counts = 0, 0, 0
            self.model.train()

            tmp_print = f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val F1':^9} | {'Val Precision':^9} | {'Val Recall':^9} | {'Elapsed':^9}"
            tmp_print += '\n' + '-' * 100
            
            if self.args.rank==0:
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

                self.model.backward(loss)
                self.model.step()

                ########################################################################################
                # Print the loss values and time elapsed for every X batches
                ########################################################################################
                if self.args.rank==0 and ((step % self.args.logging_steps == 0 and step != 0) or (step == len(train_dataloader) - 1)):
                    time_elapsed = time.time() - t0_batch
                    tmp_print = f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9.2} | {'-':^10.6} | {'-':^10.6} | {'-':^10.6} | {time_elapsed:^9.2f}"
                    print(tmp_print)
                    print(tmp_print, file=log)
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # train의 평균 loss 계산
            avg_train_loss = total_loss / len(train_dataloader)
            if self.args.rank==0:
                print("-" * 100)
                print("-" * 100, file=log)

            if valid_dataloader:
                result = self.evaluate(valid_dataloader)
                result.update({"train_loss":round(avg_train_loss, 4)})
                if self.args.rank==0:
                    print(result)
                    print(result, file=log)
                if result["loss"] < loss_check:
                    # loss가 줄어들었을 때 모델 저장
                    best_log = result
                    loss_check = result["loss"]
                    stop_count = 0
                    self.save_model()
                    #self.model.save_checkpoint(self.dir_id) #tag
                else:
                    stop_count += 1

                if stop_count >= self.args.max_stop_number:
                    if self.args.rank==0:
                        print("EARLY STOPPED")
                        print("EARLY STOPPED", file=log)
                        log.close()
                    break
            else:
                #self.model.save_checkpoint(self.dir_id) #tag
                self.save_model()

            time_elapsed = time.time() - t0_epoch
            if self.args.rank==0:
                print("Time elapsed = ", round(time_elapsed, 4))
        
        if self.args.rank==0:
            print()
            print("MODEL TRAINING END")
            print(best_log)
            print("MODEL TRAINING END", file=log)
            print(best_log, file=log)
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