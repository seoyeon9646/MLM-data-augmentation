import copy
import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

def mask_tokens(tokenizer, input_ids:torch.Tensor, mlm_prob=0.15):
    """
    Copied from huggingface/transformers/data/data_collator - torch.mask_tokens()
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = input_ids.clone()

    # mlm_probability은 15%로 BERT에섯 사용하는 확률
    probability_matrix = torch.full(labels.shape, mlm_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return input_ids, labels


def load_tuned_model(args):
    model = AutoModelForMaskedLM.from_pretrained("beomi/KcELECTRA-base")
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

    model.load_state_dict(
        torch.load(args.tuned_model_path, map_location="cuda:{}".format(args.dev_num))["module"]
    )

    return model, tokenizer


def tokenize(tokenizer, sent):
    encoded_dict = tokenizer(
        sent,
        add_special_tokens = True,
        return_attention_mask = True,
        return_tensors = "pt"
    )
    input_id, attention_mask = encoded_dict.input_ids, encoded_dict.attention_mask
    masked_input_id, label = mask_tokens(tokenizer, input_id)

    trial = 1
    while masked_input_id[0].tolist().count(tokenizer.mask_token_id)<1:
        masked_input_id, label = mask_tokens(tokenizer, input_id)
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

def mask_filling(model, tokenizer, sent, k=5, threshold=0.95):
    model.eval()

    masked_input_id, attention_mask, label = tokenize(tokenizer, sent)
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



if __name__ == "__main__":
    import argparse
    import random
    import pandas as pd
    import tqdm

    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned_model_path", type=str, help="Finetuned model path")
    parser.add_argument("--dev_num", default=0, type=int, help="cuda device number")

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

        augmented = mask_filling(model, tokenizer, line, threshold=0.9)
        while augmented==line:
            augmented = mask_filling(model, tokenizer, line, threshold=0.9)
        augmented_res.append(augmented)
        if i%50 == 0:
            print(line)
            print(augmented)

    df["augmented"] = augmented_res
    df.to_csv("../data/train_binary.csv", index=False)
    