# coding=utf-8
# Copyright (c) 2024 Ant Group

import os
import sys
import logging
import time
import datetime
import random
import math
import shutil
import argparse
import wandb
import torch
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# model
from model.fast_parser import TransformerParser
from model.fast_r2d2_insideoutside_new import FastR2D2Plus
# dataset
from reader.lazy_loader import LazyLoader
from reader.dataset import CharGPT2Dataset
from reader.gpt2_reader import SamplingByLengthDataset, MorphSegWordDataset
from reader.data_collator import CharacterCollator
# utils
from utils.tokenizer import CharLevelTokenizer, MorphCharLevelTokenizer
from utils.model_loader import get_max_epoch_step, load_checkpoint, load_model
from utils.tree_utils import get_token_tree, get_tree_from_merge_trajectory

import random
import numpy as np
import torch
from torch.utils.data import DistributedSampler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer(object):
    def __init__(self,
                 model,
                 is_master,
                 tokenizer,
                 device,
                 logger,
                 distributed=False,
                 scaler=None,
                 ext_vocab=None,
                 wandb_project=None,
                 valid_loader=None,  # Added for validation
                 early_stopping_patience=10  # Added for early stopping
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.ext_vocab = ext_vocab
        self.is_master = is_master
        self.logger = logger
        self.scaler = scaler
        self.distributed = distributed
        self.device = device
        nowtime =  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.wandb = wandb_project
        if self.wandb:
            wandb.init(project=wandb_project, name=nowtime)
            
        # Validation and Early Stopping
        self.valid_loader = valid_loader
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def train(
            self,
            data_loader: DataLoader,
            optimizer,
            scheduler,
            output_dir,
            log_step,
            epochs,
            max_grad_norm=1.0,
            max_recover_epoch=-1,
            max_recover_step=-1,
            save_steps=2000,
            no_lmloss=False,
            gradient_accumulation_steps=1
    ):
        train_iterator = trange(0, int(epochs), desc="Epoch", disable=(not self.is_master))
        total_step = len(data_loader)
        self.model.train()
        for epoch in train_iterator:
            if epoch < max_recover_epoch:
                continue
            
            epoch_iterator = tqdm(data_loader, desc="Iteration", disable=(not self.is_master))
            for step, inputs_pair in enumerate(epoch_iterator):

                if step <= max_recover_step//2:
                    continue
                max_recover_step = -1
                
                model_inputs, parser_inputs = inputs_pair
                for inputs in inputs_pair:
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)
                with torch.cuda.amp.autocast():
                    try:
                        results = self.model(parser_inputs, **model_inputs, parser_noise=1.0)
                    except RuntimeError as e:
                        print("error as", e)
                
                char_loss = results['lm_loss']
                word_loss = results['word_loss']
                span_loss = results['span_loss']
                parser_loss = results['parser_loss']
                total_loss =  char_loss + word_loss + span_loss + parser_loss
                
                if gradient_accumulation_steps > 1:
                    total_loss = total_loss / gradient_accumulation_steps

                try:
                    self.scaler.scale(total_loss).backward()

                    if (step + 1) % gradient_accumulation_steps == 0:
                        self.scaler.unscale_(optimizer)
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        scheduler.step()
                except RuntimeError as e:
                    self.logger.error(e)
                finally:
                    if (step + 1) % gradient_accumulation_steps == 0:
                        optimizer.zero_grad()

                if step % log_step == 0 and step > 0:
                    char_loss = char_loss.item() if isinstance(char_loss, torch.Tensor) else char_loss
                    span_loss = span_loss.item() if isinstance(span_loss, torch.Tensor) else span_loss
                    word_loss = word_loss.item() if isinstance(word_loss, torch.Tensor) else word_loss
                    parser_loss = parser_loss.item() if isinstance(parser_loss, torch.Tensor) else parser_loss
                    with torch.no_grad():
                        self.model.eval()
                        if self.is_master and step % (log_step*5) == 0:
                            self.print_sampled_trees(model_inputs, sample_num=5)
                        self.model.train()
                        seq_len = len(model_inputs["input_ids"][0])
                        self.logger.info(
                            f"progress: epoch{epoch} {step}/{total_step}, input_len: {seq_len}\n"
                            f"char loss: {char_loss}, parser loss: {parser_loss} "
                            f"word_loss: {word_loss}, span loss: {span_loss}"
                        )
                        if self.wandb:
                            wandb.log({"step": step, "parser loss": parser_loss, "word loss": word_loss, "span loss": span_loss})
                if step % save_steps == 0 and step > 0:
                    torch.save(self.model.state_dict(),
                           os.path.join(output_dir, f"model{epoch}_{step}.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer{epoch}_{step}.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler{epoch}_{step}.pt"))
                    if self.scaler is not None:
                        torch.save(self.scaler.state_dict(), os.path.join(output_dir, f'scaler{epoch}_{step}.pt'))
            if self.is_master:
                torch.save(self.model.state_dict(),
                           os.path.join(output_dir, f"model{epoch}.bin"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler.pt"))
                if self.scaler is not None:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, f'scaler.pt'))
    
            # Perform validation after each epoch
            if self.valid_loader is not None:
                val_loss = self.evaluate(self.valid_loader)
                self.logger.info(f"Epoch {epoch}: Validation Loss: {val_loss}")
                if self.wandb:
                    wandb.log({"epoch": epoch, "validation_loss": val_loss})
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                    # Optionally, save the best model
                    torch.save(self.model.state_dict(),
                               os.path.join(output_dir, f"best_model.bin"))
                else:
                    self.early_stop_counter += 1
                    self.logger.info(f"No improvement in validation loss for {self.early_stop_counter} epoch(s).")
                    if self.early_stop_counter >= self.early_stopping_patience:
                        self.logger.info("Early stopping triggered.")
                        if self.is_master:
                            # Load the best model before exiting
                            self.model.load_state_dict(torch.load(os.path.join(output_dir, f"best_model.bin")))
                        return  # Exit the training loop


        if self.is_master:
            torch.save(self.model.state_dict(),
                       os.path.join(output_dir, f"model.bin"))
            if self.wandb:
                wandb.finish()
    def evaluate(self, data_loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        total_steps = len(data_loader)
        with torch.no_grad():
            for inputs_pair in tqdm(data_loader, desc="Validation", disable=(not self.is_master)):
                model_inputs, parser_inputs = inputs_pair
                # Move inputs to device
                for inputs in inputs_pair:
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)

                results = self.model(parser_inputs, **model_inputs, parser_noise=1.0)
                
                char_loss = results['lm_loss']
                word_loss = results['word_loss']
                span_loss = results['span_loss']
                parser_loss = results['parser_loss']
                total_loss += (char_loss + word_loss + span_loss + parser_loss).item()
        
        avg_loss = total_loss / total_steps
        return avg_loss

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
                     default=8,
                     type=int,
                     help="training batch size")
    cmd.add_argument("--batch_max_len",
                    default=75000,
                    type=int,
                    help="training batch max tokens num")
    cmd.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before "
             "performing a backward/update pass.",
    )
    cmd.add_argument("--max_grad_norm",
                     default=1.0,
                     type=float,
                     help="Max gradient norm.")
    cmd.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    cmd.add_argument("--parser_lr", default=1e-2, type=float, help="learning rate")
    cmd.add_argument("--config_path",
                     required=True,
                     type=str,
                     help="r2d2 model config")
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
    cmd.add_argument("--local_rank",
                     default=-1,
                     type=int,
                     help="multi gpu training")
    cmd.add_argument("--pretrain_dir", default=None, type=str)
    cmd.add_argument("--epochs", default=10, type=int, help="training epochs")
    cmd.add_argument("--constraints", default='none', choices=['none', 'mat'], type=str)
    cmd.add_argument("--warm_up", type=float, default=0.1)
    cmd.add_argument("--log_step", default=100, type=int)
    cmd.add_argument("--save_steps", default=10000, type=int)
    cmd.add_argument("--sampling_times", type=int, default=0,help="sampling number/dataset size")
    cmd.add_argument("--num_samples", type=int, default=None, help="sampling number")
    cmd.add_argument("--random_sample", action='store_true', default=False)
    cmd.add_argument("--coeff_decline", type=float, default=0.1, help="parser noise decay")
    cmd.add_argument("--seed", default=42, type=int)
    cmd.add_argument("--tie_decoder", action='store_true', default=False)
    cmd.add_argument("--wandb_project", type=str, default=None)

    # vocab-pretraining specific
    cmd.add_argument("--ext_vocab_path", type=str, default=None)
    cmd.add_argument("--preload_vocab_path", type=str, default=None)
    cmd.add_argument("--span_objective", type=str, default='allspans', choices=['allspans', 'topdown'])
    cmd.add_argument("--no_continuing_prefix", action='store_true', default=False)
    cmd.add_argument("--inside_only", action='store_true', default=False)
    cmd.add_argument("--eval", action='store_true', default=False)
    cmd.add_argument("--uncased", action='store_true', default=False)

    # # ablation study
    cmd.add_argument("--no_gpt", action='store_true', default=False)
    cmd.add_argument("--no_span_emb", action='store_true', default=False)
    cmd.add_argument("--no_charloss", action='store_true', default=False)
    cmd.add_argument("--no_spanloss", action='store_true', default=False)
    cmd.add_argument("--no_wordloss", action='store_true', default=False)
    cmd.add_argument("--no_spanweights", action='store_true', default=False)

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

    # Set modelwise hyperparams
    config = AutoConfig.from_pretrained(args.config_path)
    config.update({'span_objective': args.span_objective})
    config.update({'tie_decoder': args.tie_decoder})
    config.update({'no_span_emb': args.no_span_emb})
    config.update({'inside_only': args.inside_only})
    config.update({'no_charloss': args.no_charloss})
    config.update({'no_spanloss': args.no_spanloss})
    config.update({'no_wordloss': args.no_wordloss})
    config.update({'no_spanweights': args.no_spanweights})
    config.update({'gpt': not args.no_gpt})
    
    # Init char-level tokenizer 
    tokenizer = CharLevelTokenizer(uncased=args.uncased)
    print("uncased?", args.uncased)
    if args.input_type == 'txt' and args.preload_vocab_path is None:
        tokenizer.train(args.corpus_path, output_path=os.path.join(args.output_dir,f"vocab.txt"))
    else: 
        assert args.preload_vocab_path is not None
        tokenizer.load_from_vocab_file(args.preload_vocab_path, output_path=os.path.join(args.output_dir,f"vocab.txt"), is_master=is_master)
    input_vocab_size = len(tokenizer.vocab2idx)
    config_dict = {'vocab_size': input_vocab_size}
    config.update(config_dict)

    # Prepare ext vocab
    if args.ext_vocab_path is not None:
        f_ext_vocab = open(args.ext_vocab_path)
        ext_vocab_size = 0
        ext_vocab_id_path = args.ext_vocab_path+".ids"
        # Prepare span tokenizer vocab: binarized ngram vocab
        f_ext_vocab_id = open(args.ext_vocab_path+".ids", "w") # BUG (fixed): nothing written # FIXED: f.close()
        for _, l in enumerate(f_ext_vocab.readlines()):
            ext_vocab_size += 1
            tokens = tokenizer.tokenize(l.strip()) # str -> List[str]
            token_ids = tokenizer.convert_tokens_to_ids(tokens, tensor=False)
            token_ids = [str(i) for i in token_ids]
            f_ext_vocab_id.write(",".join(token_ids)+"\n")
            # print(",".join(token_ids))
        # Char-level tokenizer vocab included (concatenated)
        f_ext_vocab_id.close()
        # Write f_ext_vocab to output_dir
        shutil.copy(args.ext_vocab_path, os.path.join(args.output_dir, "ext_vocab.txt"))
        shutil.copy(ext_vocab_id_path, os.path.join(args.output_dir, "ext_vocab.txt.ids"))
        
        pred_vocab_size = ext_vocab_size
        print("ext vocab size:", pred_vocab_size)
        config_dict = {'ext_vocab_size': pred_vocab_size}
        config.update(config_dict)
    else:
        config_dict = {'ext_vocab_size': 0}
        config.update(config_dict)

    # Output current config
    output_config_path = os.path.join(args.output_dir,f"all_config.json")
    os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
    with open(output_config_path, 'w') as config_file:
        config_file.write(config.to_json_string())

    print(f"initialize model on {global_rank}")
    set_seed(args.seed)

    # Load model and ckpts
    model = FastR2D2Plus(config)
    max_epoch, max_step = get_max_epoch_step(args.output_dir, 'model*_*.bin')
    if args.pretrain_dir is not None:
        if os.path.exists(os.path.join(args.pretrain_dir, f'model.bin')):
            model.from_pretrain(os.path.join(args.pretrain_dir, f'model.bin'))
        else:
            logging.warn('no model.bin in pretrain dir')
    elif max_epoch >= 0:
        print(f"load from checkpoint, turn: {max_epoch}_{max_step}")
        model.from_pretrain(os.path.join(args.output_dir, f'model{max_epoch}_{max_step}.bin'))
    print(f"move model to gpu:{global_rank}")
    model.to(device)

    # Load datasets
    if args.ext_vocab_path:
        external_vocab_path = args.ext_vocab_path+".ids"
    else:
        external_vocab_path = None
    print(f"start loading dataset on {global_rank}")
    if args.input_type == "txt":
        # print("max seq len", args.max_seq_len)
        dataset = SamplingByLengthDataset(
            args.corpus_path,
            max_seq_len=args.max_seq_len,
            min_len=args.min_len,
            max_line=args.max_line,
            cache_dir=args.cache_dir,
            external_vocab_path=external_vocab_path,
            sampling_times=args.sampling_times
        )
    elif args.input_type == "ids":
        lazy_loader = LazyLoader(args.corpus_path, is_array=True)
        dataset = CharGPT2Dataset(lazy_loader, tokenizer, num_samples=args.num_samples, max_seq_len=args.max_seq_len)
    else:
        raise NotImplementedError
    print("dataset len",len(dataset))
    
    # Prepare data collator
    suffix_offset = 0 if args.no_continuing_prefix else config.ext_vocab_size
    collator = CharacterCollator(tokenizer, external_vocab_path=external_vocab_path, suffix_offset=suffix_offset)
    collate_fn = collator.lm_collate_fn

    # Init Optimizer
    loader_batch_size = args.batch_size
    parser_params = []
    model_params = []
    for name, params in model.named_parameters():
        if name.find('parser.') > -1:
            parser_params.append(params)
        else:
            model_params.append(params)
    optimizer = AdamW([{"params": model_params},
                           {"params": parser_params, "lr": args.parser_lr}],
                           lr=args.lr, correct_bias=False)
    
    # Prepare Dataloader
    if global_rank == -1:
        if args.random_sample:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=loader_batch_size,
                                sampler=sampler,
                                collate_fn=collate_fn)
    elif global_rank >= 0:
        if args.eval:
            dataloader = DataLoader(
                dataset,
                batch_size=loader_batch_size,
                sampler=SequentialSampler(dataset),
                collate_fn=collate_fn,
                num_workers=1,
                drop_last=True
            )
        else: # ddp train
            print(f"initialize ddp on {global_rank}")
            dataloader = DataLoader(
                dataset,
                batch_size=loader_batch_size,
                sampler=DistributedSampler(dataset, shuffle=args.random_sample),
                collate_fn=collate_fn,
                num_workers=2//args.gradient_accumulation_steps,
                drop_last=True
            )
            model = DDP(model)#, find_unused_parameters=True)
    print(f'data total len {len(dataloader)}')

    # Prepare Scaler and Scheduler
    t_total = int(len(dataloader) * args.epochs)
    warm_up_steps = max(10, args.warm_up * t_total)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warm_up_steps,
                                                num_training_steps=t_total//args.gradient_accumulation_steps)
    # Prepare logger
    if is_master:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            shutil.copyfile(args.config_path,
                            os.path.join(args.output_dir, 'config.json'))
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

    # Retrieve ckpts
    if max_epoch >= 0:
        try:
            modules = [optimizer, scheduler, scaler]
            files = [f'optimizer{max_epoch}_{max_step}.pt', f'scheduler{max_epoch}_{max_step}.pt', \
                    f'scaler{max_epoch}_{max_step}.pt']
            load_checkpoint(modules, files, args.output_dir)
        except:
            logging.warning('load optimizer error')
            pass
    
    # Init Trainer
    trainer = Trainer(
        model,
        device=device,
        tokenizer=tokenizer,
        logger=logger,
        is_master=is_master,
        scaler=scaler,
        distributed=local_rank >= 0,
        wandb_project=args.wandb_project,
    )

    # Train or Eval
    if args.eval:
        trainer.eval(dataloader,
                    log_step=args.log_step)
    else:
        trainer.train(
            dataloader,
            optimizer,
            scheduler,
            log_step=args.log_step,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_grad_norm=args.max_grad_norm,
            max_recover_epoch=max_epoch,
            max_recover_step=max_step,
            save_steps=args.save_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )