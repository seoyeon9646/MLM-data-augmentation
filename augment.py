import copy
import torch
import tqdm
import argparse

from utils import mask_tokens

from typing import Union
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_tuned_model(args:argparse.Namespace):
    if (torch.cuda.is_available()) and (args.dev_num>=0) and (args.dev_num < torch.cuda.device_count()):
        dev = "cuda:{}".format(args.dev_num)
    else:
        dev = "cpu"
        
    model = AutoModelForMaskedLM.from_pretrained(args.tuned_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tuned_model_path)

    model.to(dev)
    return model, tokenizer, dev

def tokenize(tokenizer:AutoTokenizer, sent:str):
    encoded_dict = tokenizer(
        sent,
        add_special_tokens = True,
        return_attention_mask = True,
        return_tensors = "pt"
    )
    input_id, attention_mask = encoded_dict.input_ids, encoded_dict.attention_mask

    return input_id, attention_mask

def is_same_token_type(org_token:str, candidate:str) -> bool:
    '''
    후보 필터링 조건을 만족하는지 확인
    - 후보와 원 토큰의 타입을 문장부호와 일반 토큰으로 나누어 같은 타입에 속하는지 확인
    '''
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

def candidate_filtering(tokenizer:AutoTokenizer,
                        input_ids:list,
                        idx:int,
                        org:int,
                        candidates:Union[list, torch.Tensor]) -> int:
    '''
    후보 필터링 조건에 만족하는 최적의 후보 선택
    1. 원래 토큰과 후보 토큰이 같은 타입(is_same_token_type 참고)
    2. 현 위치 앞 혹은 뒤에 동일한 토큰이 있지 않음
    '''

    org_token = tokenizer.convert_ids_to_tokens([org])[0]
    candidate_tokens = tokenizer.convert_ids_to_tokens(candidates.cpu().tolist())

    for rank, token in enumerate(candidate_tokens):
        if org_token!=token and is_same_token_type(org_token, token):
            if input_ids[idx-1]==candidates[rank] or input_ids[idx+1]==candidate_tokens[rank]:
                continue
            return candidates[rank]

    return org

def augment_one_sent(model:AutoModelForMaskedLM,
                    tokenizer:AutoTokenizer,
                    sent:str,
                    dev:Union[str, torch.device],
                    args:Union[argparse.Namespace, dict]) -> str:
    '''
    한 문장에 랜덤으로 마스킹을 적용하여 새로운 문장을 생성(증강)

    args:
        model(AutoModelForMaskedLM)     : finetuned model
        tokenizer(AutoTokenizer)
        sent(str)                       : 증강할 문장
        dev(str or torch.device)
        args(argparse.Namespace)
            - k(int, default=5) : 사용할 후보의 개수. k개의 후보 적절한 토큰이 없을 경우 원래 토큰 그대로 유지
            - threshold(float, default=0.95) : 확률 필터링에 사용할 임계치.
                                               마스크에 대해서 특정 후보 토큰을 생성할 확률이 임계치보다 클 경우에는 별도의 필터링 없이 후보를 그대로 사용.
           -  mlm_prob(float, default=0.15) : 마스킹 비율
        
    return:
        (str) : 증강 문장
    '''

    if type(args) == argparse.Namespace:
        k = args.k
        threshold = args.threshold
        mlm_prob = args.mlm_prob
    else:
        ## type == dict
        k = args["k"]
        threshold = args["threshold"]
        mlm_prob = args["mlm_prob"]

    model.eval()

    input_id, attention_mask  = tokenize(tokenizer, sent)
    org_ids = copy.deepcopy(input_id[0])
    
    masked_input_id, _ = mask_tokens(tokenizer, input_id, mlm_prob, do_rep_random=False)
    while masked_input_id.cpu().tolist()[0].count(tokenizer.mask_token_id) < 1:
        masked_input_id, _ = mask_tokens(tokenizer, input_id, mlm_prob, do_rep_random=False)
    
    with torch.no_grad():
        masked_input_id, attention_mask = masked_input_id.to(dev), attention_mask.to(dev)
        output = model(masked_input_id, attention_mask = attention_mask)
        logits = output["logits"][0]

    copied = copy.deepcopy(masked_input_id.cpu().tolist()[0])
    for i in range(len(copied)):
        if copied[i] == tokenizer.mask_token_id:
            org_token = org_ids[i]
            prob = logits[i].softmax(dim=0)
            probability, candidates = prob.topk(k)
            if probability[0]<threshold:
                res = candidate_filtering(tokenizer, copied, i, org_token, candidates)
            else:
                res = candidates[0]
            copied[i] = res

    copied = tokenizer.decode(copied, skip_special_tokens=True)

    return copied


def batch_augment(model:AutoModelForMaskedLM,
                tokenizer:AutoTokenizer,
                dataset:torch.utils.data.Dataset,
                dev:Union[str, torch.device],
                args:argparse.Namespace) -> str:
    '''
    배치 단위의 문장에 랜덤으로 마스킹을 적용하여 새로운 문장 배치를 생성(증강)

    args:
        model(AutoModelForMaskedLM)
        tokenizer(AutoTokenizer)
        dataset(torch.utils.data.Dataset)
        dev(str or torch.device)
        args(argparse.Namespace)
            - k(int, default=5)
            - threshold(float, default=0.95)
           -  mlm_prob(float, default=0.15)
        
    return:
        (list) : 증강한 문장들의 리스트
    '''

    k = args.k
    threshold = args.threshold
    mlm_prob = args.mlm_prob
    batch_size = args.batch_size

    model.eval()

    augmented_res = []
    dataloader = DataLoader(dataset, batch_size = batch_size)
    for batch in tqdm.tqdm(dataloader):
        #########################################################
        # 인풋 문장에 랜덤으로 마스킹 적용
        input_ids, attention_masks = batch[0], batch[1]
        masked_input_ids, _ = mask_tokens(tokenizer, input_ids, mlm_prob, do_rep_random=False)

        masked_input_ids = masked_input_ids.to(dev)
        attention_masks = attention_masks.to(dev)
        labels = input_ids
        #########################################################

        with torch.no_grad():
            output = model(masked_input_ids, attention_mask = attention_masks)
            logits1 = output["logits"]

        #########################################################
        # 배치 내의 문장 별로 후보 필터링을 적용하고, 결과를 토대로 새로운 문장 생성
        augmented1 = []
        for sent_no in range(len(masked_input_ids)):
            copied = copy.deepcopy(input_ids.cpu().tolist()[sent_no])

            for i in range(len(masked_input_ids[sent_no])):
                if masked_input_ids[sent_no][i] == tokenizer.pad_token_id:
                    break

                if masked_input_ids[sent_no][i] == tokenizer.mask_token_id:
                    org_token = labels.cpu().tolist()[sent_no][i]
                    prob = logits1[sent_no][i].softmax(dim=0)
                    probability, candidates = prob.topk(k)
                    if probability[0]<threshold:
                        res = candidate_filtering(tokenizer, copied, i, org_token, candidates)
                    else:
                        res = candidates[0]
                    copied[i] = res

            copied = tokenizer.decode(copied, skip_special_tokens=True)
            augmented1.append(copied)
        #########################################################
        augmented_res.extend(augmented1)

    return augmented_res


if __name__ == "__main__":
    import random

    from data_loader import AugmentDataSet

    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned_model_path", default="seoyeon96/KcELECTRA-MLM", type=str, help="Finetuned model path")
    parser.add_argument("--dev_num", default=-1, type=int, help="cuda device number")
    parser.add_argument("--input_file", default=None, type=str)   # 증강을 적용할 데이터
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--mlm_prob", default=0.15, type=float)
    parser.add_argument("--threshold", default=0.95, type=float)
    parser.add_argument("--k", default=5, type=int)

    args = parser.parse_args()
    model, tokenizer, dev = load_tuned_model(args) 

    if args.batch_size > 1:
        if args.input_file is None:
            raise Exception("input_file is None")
        
        with open(args.input_file, "r") as f:
            corpus = f.readlines()

        dataset = AugmentDataSet(corpus, tokenizer)
        augmented = batch_augment(model, tokenizer, dataset, dev, args)
    else:
        while True:
            input_sen = input("INPUT = ").strip()
            augmented = augment_one_sent(model, tokenizer, input_sen, dev, args)
            print("OUTPUT = ", augmented)
            print("-"*30)
    