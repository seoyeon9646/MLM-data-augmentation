import copy
import torch

from utils import mask_tokens
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_tuned_model(args):
    model = AutoModelForMaskedLM.from_pretrained("beomi/KcELECTRA-base")
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

    model.load_state_dict(
        torch.load(args.tuned_model_path, map_location="cuda:{}".format(args.dev_num))["module"]
    )

    return model, tokenizer


def tokenize(tokenizer, sent, mlm_prob):
    encoded_dict = tokenizer(
        sent,
        add_special_tokens = True,
        return_attention_mask = True,
        return_tensors = "pt"
    )
    input_id, attention_mask = encoded_dict.input_ids, encoded_dict.attention_mask
    masked_input_id, label = mask_tokens(tokenizer, input_id, mlm_prob, do_rep_random=False)

    trial = 1
    while masked_input_id[0].tolist().count(tokenizer.mask_token_id)<1:
        masked_input_id, label = mask_tokens(tokenizer, input_id, mlm_prob, do_rep_random=False)
        trial +=1
        if trial>5:
            break

    return masked_input_id, attention_mask, label

def is_same_token_type(org_token, candidate):
    res = False
    if org_token[0]=="#" and org_token[2:].isalpha()==candidate.isalpha():
        res = True
    elif candidate[0]=="#" and org_token.isalpha()==candidate[2:].isalpha():
        res = True
    elif candidate[0]=="#" and org_token[0]=="#" and org_token[2:].isalpha()==candidate[2:].isalpha():
        res = True
    elif org_token.isalpha()==candidate.isalpha() and (candidate[0]!="#" and org_token[0]!="#"):
        res = True

    return res

def candidate_filtering(tokenizer, input_ids, idx, org, candidates):
    org_token = tokenizer.convert_ids_to_tokens([org])[0]
    candidate_tokens = tokenizer.convert_ids_to_tokens(candidates)

    for rank, token in enumerate(candidate_tokens):
        if org_token!=token and is_same_token_type(org_token, token):
            if input_ids[idx-1]==candidates[rank] or input_ids[idx+1]==candidate_tokens[rank]:
                continue
            return candidates[rank]

    return org

def mask_filling(model, tokenizer, sent, k=5, threshold=0.95, mlm_prob=0.15):
    model.eval()

    masked_input_id, attention_mask, label = tokenize(tokenizer, sent, mlm_prob)
    with torch.no_grad():
        output = model(masked_input_id, attention_mask = attention_mask)
        logits = output["logits"]

    copied = copy.deepcopy(masked_input_id[0])
    top1_list = []
    for i in range(len(masked_input_id[0])):
        org_token = label[0][i]
        prob = logits[0][i].softmax(dim=0)
        probability, candidates = prob.topk(k)
        if masked_input_id[0][i] == tokenizer.mask_token_id:
            probability, candidates = prob.topk(k)
            if probability[0]<threshold:
                res = candidate_filtering(tokenizer, copied, i, org_token, candidates)
            else:
                res = candidates[0]
            copied[i] = res
        top1_list.append(candidates[0])

    copied = tokenizer.decode(copied, skip_special_tokens=True)

    return copied

def tokenize_batch(tokenizer, batch_sent, mlm_prob, max_length=100):
    input_ids, attention_masks = [], []
    for sent in batch_sent:
        encoded_dict = tokenizer(
            sent,
            add_special_tokens =True,
            max_length = max_length,
            padding = "max_length",
            truncation = True,
            return_attention_mask = True,
            return_tensors="pt"
        )
        input_ids.append(encoded_dict.input_ids)
        attention_masks.append(encoded_dict.attention_mask)

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    masked_input_id, labels = mask_tokens(tokenizer, input_ids, mlm_prob, do_rep_random=False)

    return masked_input_id, attention_masks, labels

def batch_augment(model, tokenizer, batch_sent, k=5,  threshold=0.95, mlm_prob=0.15):
    model.eval()

    masked_input_ids, attention_masks, labels = tokenize_batch(tokenizer, batch_sent, mlm_prob)
    with torch.no_grad():
        output = model(masked_input_ids, attention_mask = attention_masks)
        logits = output["logits"]

    augmented = []
    for sent_no in range(len(logits)):
        copied = copy.deepcopy(masked_input_ids[sent_no])
        for i in range(len(masked_input_ids[sent_no])):
            org_token = labels[sent_no][i]
            prob = logits[0][i].softmax(dim=0)
            probability, candidates = prob.topk(k)
            if masked_input_ids[sent_no][i] == tokenizer.mask_token_id:
                probability, candidates = prob.topk(k)
                if probability[0]<threshold:
                    res = candidate_filtering(tokenizer, copied, i, org_token, candidates)
                else:
                    res = candidates[0]
                copied[i] = res
        copied = tokenizer.decode(copied, skip_special_tokens=True)
        augmented.append(copied)

    return augmented



if __name__ == "__main__":
    import argparse
    import random
    import pandas as pd
    import tqdm

    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned_model_path", type=str, help="Finetuned model path")
    parser.add_argument("--dev_num", default=0, type=int, help="cuda device number")
    parser.add_argument("--mlm_prob", default=0.15, type=float, help="cuda device number")

    args = parser.parse_args()
    model, tokenizer = load_tuned_model(args) 
    '''
    while True:
        sent = input("input sentence = ")
        augmented, _, top1_sent = mask_filling(model, tokenizer, sent)
        print(augmented)
        print(top1_sent)
        print()
    '''
    df = pd.read_csv("../data/train_binary.csv")
    augmented_res = []

    for i in tqdm.tqdm(range(len(df))):
        line = df["comments"][i].replace("\n", "")

        augmented = mask_filling(model, tokenizer, line, mlm_prob=args.mlm_prob)
        while augmented==line:
            augmented = mask_filling(model, tokenizer, line, threshold=0.9, mlm_prob=args.mlm_prob)
        augmented_res.append(augmented)
        if i%50 == 0:
            print(line)
            print(augmented)

    ndf = pd.DataFrame({
        "comments":df["comments"].tolist(),
        "augmented":augmented_res
    })
    df.to_csv("../data/train_binary{}.csv".format(str(args.mlm_prob)), index=False)
    
    '''
    f = open("./data/raw_data.txt", "r")
    corpus = f.readlines()
    random.shuffle(corpus)

    total_result = []
    for line in tqdm.tqdm(corpus):
        line = line.replace("\n", "")

        augmented1 = mask_filling(model, tokenizer, line, threshold=0.9)
        augmented2 = mask_filling(model, tokenizer, line, threshold=0.9)

        if line!=augmented1 and augmented1!=augmented2 and line!=augmented2:
            total_result.append({"comments":line, "augmented1":augmented1, "augmented2":augmented2})

        if len(total_result)>=10000:
            break

    df = pd.DataFrame(total_result)
    df.to_csv("../data/train_unlabeled10k.csv", index=False)
    '''