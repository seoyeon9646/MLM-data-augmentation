import torch
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from argparse import Namespace

class DataSet(Dataset):
    def __init__(self, df:pd.DataFrame, tokenizer:AutoTokenizer, args:Namespace):
        self.data = df.to_dict("records")
        input_ids, attention_masks = [], []
            
        for line in tqdm(self.data):
            try:
                comments = line["comments"].replace("\n", "")
                encoded_dict = tokenizer(
                    comments,
                    add_special_tokens =True,
                    max_length = args.max_seq_len,
                    padding = "max_length",
                    truncation = True,
                    return_attention_mask = True,
                    return_tensors="pt"
                )
                input_ids.append(encoded_dict.input_ids)
                attention_masks.append(encoded_dict.attention_mask)
            except:
                continue
        
        # flattening : convert it to 0 dim torch tensor
        self.input_ids = torch.cat(input_ids, dim = 0)
        self.attention_masks = torch.cat(attention_masks, dim = 0)


    # get data length
    def __len__(self):
        return len(self.input_ids)
    
    # get each data info
    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]

        return input_id, attention_mask



class AugmentDataSet(Dataset):
    def __init__(self, sent_list:list, tokenizer:AutoTokenizer, args:Namespace):
        total_result = []
        input_ids, attention_masks = [], []
            
        for line in tqdm(sent_list):
            try:
                comments = line
                encoded_dict = tokenizer(
                    comments,
                    add_special_tokens =True,
                    max_length = 100,
                    padding = "max_length",
                    truncation = True,
                    return_attention_mask = True,
                    return_tensors="pt"
                )
                input_ids.append(encoded_dict.input_ids)
                attention_masks.append(encoded_dict.attention_mask)
                total_result.append({"comments":line})
            except:
                continue
        
        # flattening : convert it to 0 dim torch tensor
        self.input_ids = torch.cat(input_ids, dim = 0)
        self.attention_masks = torch.cat(attention_masks, dim = 0)
        self.df = pd.DataFrame(total_result)

    # get data length
    def __len__(self):
        return len(self.input_ids)
    
    # get each data info
    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]

        return input_id, attention_mask