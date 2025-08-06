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
import yaml
from easydict import EasyDict as edict

from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
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
from utils.tree_utils import get_token_tree, get_tree_from_merge_trajectory, find_span_in_tree, get_token_tree_from_splits, get_spans_from_splits
from utils.metrics import UReca

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def span_recall_in_tree(root, segmentations, output_dir=None):
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
    
def get_spans_in_tree(root):
    spans = []
    q = [root]
    while q:
        current = q.pop()
        spans.append((current.i, current.j))
        if current.i != current.j:
            q.append(current.left)
            q.append(current.right)
    return spans
 

class Trainer(object):
    def __init__(self,
                 model,
                 parser,
                 is_master,
                 tokenizer,
                 device,
                 logger,
                 n_gpu=1):
        self.model = model
        self.parser = parser
        self.tokenizer = tokenizer
        self.is_master = is_master
        self.logger = logger

        self.device = device
        self.n_gpu = n_gpu

    def eval(
            self,
            data_loader: DataLoader,
            output_dir=None,
            inference_mode='parser'
    ):
        metric_rec = UReca()
        epoch_iterator = tqdm(data_loader, desc="Iteration")
        with torch.no_grad():
            cnt = 0
            for input_pair in epoch_iterator:
                if cnt > 1:
                    break
                cnt += 1
                for inputs in input_pair:
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)
                model_inputs, parser_inputs = input_pair
                self.model.eval()
                if inference_mode=="parser":
                    merge_trajectories, _ = self.model.parser(parser_inputs["input_ids"],parser_inputs["attention_mask"],
                                                     noise_coeff=0)
                    merge_trajectories = merge_trajectories.to('cpu').data.numpy()
                    reverse_splits=False  

                elif inference_mode=="r2d2":
                    merge_trajectories = self.model.inside_inference(parser_inputs, **model_inputs)
                    merge_trajectories = merge_trajectories.to('cpu').data.numpy()
                    reverse_splits=True 
                elif inference_mode=="pcfg":
                    merge_trajectories = self.model.evaluate(model_inputs, decode_type='mbr', splits=True)['splits']

                for i, m in enumerate(merge_trajectories):
                    orig_tokens = self.tokenizer.convert_ids_to_tokens(parser_inputs["input_ids"][i].cpu().data.numpy())
                    seq_len = parser_inputs["attention_mask"][i].sum().item()
                    gold = model_inputs["labels"][i] # List[List[[st, ed]]]
                    if inference_mode == "pcfg":
                        tree_str = get_token_tree_from_splits(m, orig_tokens[:seq_len])
                        pred = get_spans_from_splits(m, orig_tokens[:seq_len])
                        print(tree_str)
                    else:
                        root, tree_str = get_tree_from_merge_trajectory(m, seq_len, orig_tokens, reverse=reverse_splits)
                        pred = get_spans_in_tree(root)
                        print(tree_str)
                    metric_rec(pred, gold, filter_leaf=False)
            print(metric_rec)

    def print_sampled_trees(self, model_inputs, sample_num=5):
        input_ids = model_inputs['input_ids']
        masks = model_inputs['masks']
        # ptrs 
        i = 0
        printed_cnt = 0
        num_words = input_ids.shape[0] # upper bound of i
        print_threshold = min(num_words, sample_num) # upper bound of printed_cnt
        # parser splits
        merge_trajectories, _ = self.model.module.parser(input_ids=input_ids, attention_mask=masks, noise_coeff=0)
        while printed_cnt < print_threshold and i < num_words:
            seq_len = masks[i].sum().item()
            if seq_len < 3:
                i += 1
                pass
            else:
                current_input_ids = model_inputs['input_ids'][i].cpu().data.numpy()
                tokens = self.tokenizer.convert_ids_to_tokens(current_input_ids)
                _, tree_str = get_tree_from_merge_trajectory(merge_trajectories[i].cpu().data.numpy(), seq_len, tokens)
                self.logger.info(
                                    f"inputs: {''.join(tokens[:seq_len])} "
                                    f"parsed tree: {tree_str} "
                                )
                printed_cnt += 1
                i += 1
if __name__ == "__main__":

    cmd = argparse.ArgumentParser("Arguments to pretrain R2D2")
    cmd.add_argument("--batch_size",
                     default=32,
                     type=int,
                     help="training batch size")
    cmd.add_argument("--config_path",
                     required=True,
                     type=str,
                     help="bert model config")
    cmd.add_argument("--vocab_dir",
                     required=False,
                     type=str,
                     help="Directory to the vocabulary")
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
    cmd.add_argument("--output_dir", required=True, type=str, help="save dir")
    cmd.add_argument("--seperator", type=str, default=None)
    cmd.add_argument("--local_rank",
                     default=-1,
                     type=int,
                     help="multi gpu training")
    cmd.add_argument("--pretrain_dir", default=None, type=str)
    cmd.add_argument("--checkpoint_name", default='model.bin', type=str)
    cmd.add_argument("--random_sample", action='store_true', default=False)
    cmd.add_argument("--transformer_parser", action='store_true', default=False)
    cmd.add_argument("--uncased", default=False, action='store_true')

    ## ext vocab only
    cmd.add_argument("--gpt", action='store_true', default=False)
    cmd.add_argument("--ext_vocab_path", default=None, type=str)
    cmd.add_argument("--suffix_offset", default=0, type=int)
    cmd.add_argument("--pure_io", action='store_true', default=False)
    cmd.add_argument("--inference_mode", type=str, default='parser', choices=['parser', 'r2d2', 'pcfg'])

    cmd.add_argument("--seed", default=42, type=int)
    ## 
    cmd.add_argument("--lang", type=str, choices=['en','fr','de'])

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
    if not os.path.exists(args.output_dir) and is_master:
        os.makedirs(args.output_dir, exist_ok=True)    
    print("Using a char-level tokenizer.")
    # train a char-level tokenizer from scratch
    tokenizer = CharLevelTokenizer(uncased=False)
    print("uncased?",args.uncased)
    if args.vocab_dir:
        tokenizer.load_from_vocab_file(vocab_path=os.path.join(args.vocab_dir,f"vocab.txt"),
                                        output_path=os.path.join(args.output_dir,f"vocab.txt"))
    else:
        assert False, "Pass vocab.txt under args.vocab_dir to initialize tokenizer"
    # vocab size is set on the fly
    input_vocab_size = len(tokenizer.vocab2idx)

    print(f"initialize model on {global_rank}")
    set_seed(args.seed)

    if args.inference_mode=="pcfg":
        from parser.model import FastTNPCFG, Simple_N_PCFG, NeuralPCFG
        yaml_cfg = yaml.load(open("config/rank_pcfg/fast_tnpcfg_r1000_nt9000_t4500_curriculum0.yaml", 'r'), Loader=yaml.Loader)
        pcfg_args = edict(yaml_cfg)
        pcfg_args = edict(yaml_cfg).model
        model = FastTNPCFG(pcfg_args, input_vocab_size)
    else:
        config = AutoConfig.from_pretrained(args.config_path)
        if args.pure_io:
            from model.fast_r2d2_insideoutside_pure import FastR2D2Plus
            model = FastR2D2Plus(config)
        else:
            from model.fast_r2d2_insideoutside_new import FastR2D2Plus
            model = FastR2D2Plus(config)
    parser = None # included in model

    max_epoch, max_step = get_max_epoch_step(args.output_dir, 'model*_*.bin')
    if args.pretrain_dir is not None:
        best_model_path = os.path.join(args.pretrain_dir, args.checkpoint_name)
        if os.path.exists(best_model_path):
            if args.inference_mode == "pcfg":
                state_dict = torch.load(best_model_path, map_location=lambda a, b: a)
                transfered_state_dict = {}
                for k, v in state_dict.items():
                    new_k = k.replace('module.', '')
                    transfered_state_dict[new_k] = v
                model.load_state_dict(transfered_state_dict, strict=True)
            else:
                model.from_pretrain(os.path.join(args.pretrain_dir, args.checkpoint_name))
        else:
            raise FileNotFoundError(f"model.bin not found in {args.pretrain_dir}/{args.checkpoint_name}")

    elif max_epoch >= 0:
        print(f"load from checkpoint, turn: {max_epoch}_{max_step}")
        model.from_pretrain(os.path.join(args.output_dir, f'model{max_epoch}_{max_step}.bin'))

    print(f"move model to gpu:{global_rank}")
    model.to(device)
    
    loader_batch_size = args.batch_size

    print(f"start loading dataset on {global_rank}")
    if '.csv' in args.corpus_path:
        dataset = CSVReader(args.corpus_path, tokenizer, args.lang)
    else:
        dataset = MorphSegReader(
            args.corpus_path,
            tokenizer,
            min_len=args.min_len,
            max_line=args.max_line
        )
    print("dataset len",len(dataset))
    print(args.ext_vocab_path)
    collator = CharacterCollator(tokenizer, external_vocab_path=args.ext_vocab_path, suffix_offset=args.suffix_offset)
    collate_fn = collator.eval_word_collate_fn

    if global_rank == -1:
        if args.random_sample:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset,
                                batch_size=loader_batch_size,
                                sampler=sampler,
                                collate_fn=collate_fn)
        print(f'data total len {len(dataloader)}')
        n_gpu = 1
        
    elif global_rank >= 0:
        n_gpu = 1
        print(f"initialize ddp on {global_rank}")
        dataloader = DataLoader(
            dataset,
            batch_size=loader_batch_size,
            sampler=DistributedSampler(dataset, shuffle=args.random_sample),
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=1
        )
        print(f'data total len {len(dataloader)}')
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = max(10, args.warm_up * t_total)
        model = DDP(model)#, find_unused_parameters=True)

    if is_master:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except RuntimeError:
            pass
    if is_master:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(args.output_dir, "training_log.txt"), mode="a", encoding="utf-8")
        ch = logging.StreamHandler(sys.stdout)
        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        logger = logging
    
    trainer = Trainer(
        model,
        parser=parser,
        device=device,
        tokenizer=tokenizer,
        logger=logger,
        is_master=is_master,
        n_gpu=n_gpu,
    )
   
    trainer.eval(
        dataloader,
        output_dir=args.output_dir,
        inference_mode=args.inference_mode
    )
