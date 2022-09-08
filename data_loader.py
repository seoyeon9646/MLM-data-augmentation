import torch
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self, df, tokenizer, args):
        self.data = df.to_dict("records") if type(df) == pd.DataFrame else df
        input_ids, attention_masks = [], []
            
        for line in tqdm(self.data):
            # 추후에 전처리 과정 추가 필요
            # dynamic padding 추가도 가능
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