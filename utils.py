import torch
import torch.nn as nn
import deepspeed

from transformers import (
  AutoTokenizer,
  AutoModelForMaskedLM,
  AdamW,
  get_scheduler
)

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

  # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
  indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
  input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

  # 10% of the time, we replace masked input tokens with random word
  indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
  random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
  input_ids[indices_random] = random_words[indices_random]


  return input_ids, labels


def load_tokenizer(args):
  return AutoTokenizer.from_pretrained(args.model_name_or_path)

    
def initialize_model(args, total_steps):
  model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

  if (torch.cuda.is_available() and not args.no_cuda):
        if (not args.multi) and (not args.distributed):
            device = "cuda:" + str(args.dev_num)
        else:
            n_dev = torch.cuda.device_count()
            dev_list = list(range(n_dev))
            model = nn.DataParallel(model, device_ids = dev_list, output_device=dev_list[0])
            device = dev_list[0]
  else:
    device = "cpu"
  model.to(device)

  optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate,
                    eps = args.eps,
                    weight_decay = args.weight_decay)

  scheduler = get_scheduler(args.scheduler_name,
                            optimizer,
                            num_warmup_steps=int(total_steps * args.warmup_proportion),
                            num_training_steps= total_steps)

  return model, optimizer, scheduler, device

def initialize_model_with_ds(args):
  model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
  model, optimizer, _, scheduler = deepspeed.initialize(model = model, args=args, model_parameters=model.parameters())

  return model, optimizer, scheduler