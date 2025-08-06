import os
import sys
import logging
import time
import random
import math
import shutil
import argparse
import torch
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoConfig, PreTrainedTokenizerFast, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# model
from model.fast_parser import TransformerParser
from model.fast_r2d2_insideoutside_new import FastR2D2Plus
# dataset
from reader.seg_reader import MorphSegReader, CSVReader
from reader.data_collator import CharacterCollator
# utils
from utils.tokenizer import CharLevelTokenizer
from utils.model_loader import get_max_epoch_step, load_checkpoint, load_model
from utils.tree_utils import get_token_tree, get_tree_from_merge_trajectory, find_span_in_tree
from utils.metrics import UReca

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def span_recall_in_tree(root, segmentations):
    """
    Compute the span recall rate in a tree 
        given (potentially) multiple gold segmentations.
    """
    max_recall = (-1,1)
    for seg in segmentations:
        cnt_hit = 0
        cnt_tgt = len(seg)
        # don't take into consideration the cases where no segmentation
        # present in gold: because root always hit the whole word
        for span in seg:
            hit = find_span_in_tree(root, span[0], span[1])
            if hit:
                cnt_hit += 1
            
        if cnt_hit / cnt_tgt > max_recall[0] / max_recall[1]:
            max_recall = (cnt_hit, cnt_tgt)
    return max_recall

class Trainer(object):
    def __init__(self,
                 is_master,
                 tokenizer,
                 device,
                 n_gpu=1):
        self.tokenizer = tokenizer
        self.is_master = is_master

        self.device = device
        self.n_gpu = n_gpu

    def eval(
            self,
            data_loader: DataLoader
    ):
        total_acc = 0
        total_F1 = 0
        normalizer = AutoTokenizer.from_pretrained('bert-base-uncased').backend_tokenizer.normalizer
        epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
        with torch.no_grad():
            cnt = 0
            hit_flag = 0
            for input_pair in epoch_iterator:
                for inputs in input_pair:
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)
                model_inputs, parser_inputs = input_pair
                B = model_inputs['input_ids'].shape[0]
                cnt += B
                
                for b in range(B):
                    pred = self.tokenizer.tokenize(model_inputs['ori_word'][b])
                    # remove all the preceding ## from the tokens
                    pred = [p.lstrip('##') for p in pred]
                    
                    golds = model_inputs['label_tokens'][b] #str
                    sample_acc = 0
                    max_sample_F1 = 0
                    for gold in golds:
                        sample_hit = 0
                        gold = gold.strip()
                        gold = normalizer.normalize_str(gold)
                        gold_segs = gold.split()
                        for tgt in gold_segs:
                            if tgt in pred:
                                sample_hit += 1
                        sample_recall = sample_hit/len(gold_segs)
                        sample_prec = sample_hit/len(pred)
                        if sample_recall + sample_prec == 0:
                            sample_F1 = 0
                        else:
                            sample_F1 = 2*sample_recall*sample_prec/(sample_recall+sample_prec)
                            if sample_F1 > max_sample_F1: # if multiple gold, take max
                                max_sample_F1 = sample_F1
                        if pred == gold_segs:
                            # hit this sample if hit any gold
                            sample_acc = 1

                    if sample_acc == 1:
                        hit_flag += 1
                    total_F1 += max_sample_F1
                    total_acc += sample_acc
                
            print(f"Validation F1 ({args.tokenizer_type}): {(total_F1/cnt):.4f}")
            print(f"Validation Accuracy ({args.tokenizer_type}): {(total_acc/cnt):.4f}")

if __name__ == "__main__":
    cmd = argparse.ArgumentParser("Arguments to pretrain R2D2")
    cmd.add_argument("--batch_size",
                     default=1,
                     type=int,
                     help="training batch size")
    cmd.add_argument("--input_type",
                     default="txt",
                     type=str,
                     choices=["txt", "ids"])
    cmd.add_argument("--corpus_path",
                     required=True,
                     type=str,
                     help="path to the training corpus")
    cmd.add_argument('--cache_dir', required=False, default=None, type=str)
    cmd.add_argument("--max_seq_len", default=512, type=int) # NOTE: max len per single item
    cmd.add_argument("--min_len", default=2, type=int)
    cmd.add_argument("--max_line", default=-1, type=int)
    cmd.add_argument("--seperator", type=str, default=None)
    cmd.add_argument("--local_rank",
                     default=-1,
                     type=int,
                     help="multi gpu training")
    cmd.add_argument("--pretrain_dir", default=None, type=str)
    cmd.add_argument("--pretrain_name", default='model.bin', type=str)
    cmd.add_argument("--random_sample", action='store_true', default=False)

    ## tokenizer-specific
    cmd.add_argument("--tokenizer_type", required=True, type=str)
    cmd.add_argument("--lang", default='en', type=str)
    cmd.add_argument("--uncase", action='store_true', default=True)
    cmd.add_argument("--tokenizer_path", required=True, type=str, 
                    help='contains pretrained model and vocab file')
    cmd.add_argument("--ext_vocab_path", default=None, type=str)
    cmd.add_argument("--pure_input_path", default=None, type=str, help='only used for training tree tokenizer')
    cmd.add_argument("--seed", default=42, type=int)

    args = cmd.parse_args(sys.argv[1:])
    
    # torchrun migration
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = -1
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)  # for multi-process in a single machine with multiple GPUs.
        global_rank = local_rank
        while True:
            try:
                torch.distributed.init_process_group(backend="nccl", init_method="env://")
                if torch.distributed.is_initialized():
                    break
            except ValueError:
                time.sleep(5)
            except Exception as e:
                logging.error(e)
                logging.error("Exit with unknown error")
                exit(-1)
        device = torch.device("cuda")
    else:
        global_rank = -1
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")

    is_master = local_rank == -1 or global_rank == 0
    
    if args.tokenizer_type == 'tree':
        from utils.tree_tokenizer import TreeTokenizer
        tokenizer_config_path = os.path.join(args.tokenizer_path, 'all_config.json')
        tokenizer = TreeTokenizer(tokenizer_config_path, args.tokenizer_path, uncased=True, continuing_prefix='')
        pure_input_path = args.pure_input_path
        tokenizer.train(dataset_path=pure_input_path, ext_vocab_path=args.ext_vocab_path,
                        batch_size=32)
    else:
        tokenizer = PreTrainedTokenizerFast(
            # tokenizer_object=tokenizer,
            tokenizer_file=args.tokenizer_path, # You can load from the tokenizer file, alternatively
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

    # vocab size is set on the fly

    print(f"initialize model on {global_rank}")
    set_seed(args.seed)
    
    loader_batch_size = args.batch_size

    print(f"start loading dataset on {global_rank}")
    if '.csv' in args.corpus_path:
        dataset = CSVReader(args.corpus_path, tokenizer, lang=args.lang, uncase=args.uncase, output_path='data/compound/valid_input.txt')
    else:
        dataset = MorphSegReader(
            args.corpus_path,
            tokenizer,
            min_len=args.min_len,
            max_line=args.max_line,
            output_path='data/morpho/goldstd_inputs.txt'
        )
    print("dataset len",len(dataset))
    collator = CharacterCollator(tokenizer)
    collate_fn = collator.eval_word_collate_fn

    dataloader = DataLoader(
        dataset,
        batch_size=loader_batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collate_fn,
        drop_last=True
    )
    print(f'data total len {len(dataloader)}')

    
    trainer = Trainer(
        device=device,
        tokenizer=tokenizer,
        is_master=is_master
    )
   
    trainer.eval(
        dataloader
    )
