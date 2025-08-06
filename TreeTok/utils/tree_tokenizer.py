import os
import math
from typing import List, Dict, Union
import numpy as np
import torch
from functools import reduce
from collections import defaultdict
from transformers import AutoConfig
from model.fast_r2d2_insideoutside_new import FastR2D2Plus
from reader.word_reader import WordDataset
from utils.vocab_builder import load_span_tokenizer
from transformers import AutoTokenizer 


def load_normalizer(name: str):
    if name == 'bert-base-uncased':
        tokenizer = AutoTokenizer.from_pretrained(name)
        return tokenizer.backend_tokenizer.normalizer
    else:
        raise NotImplementedError

class TreeTokenizer(object):
    def __init__(self, config_path, model_path, uncased=False, continuing_prefix=''):
        """
        :param ext_vocab_path: via vocab mining
        """
        self.model = self.load_pretrained_tokenizer(config_path, model_path)
        self.char_tokenizer  = self.load_basic_char_tokenizer(model_path, uncased=uncased)
        if uncased:
            self.normalizer = load_normalizer('bert-base-uncased')
        else:
            self.normalizer = None
        self.continuing_prefix = continuing_prefix
        self.vocab2idx = dict()
        self.idx2vocab = dict()
        self.vocab2seg = dict()
        self.vocab2score = defaultdict(float)
        self.PAD = "[PAD]"
        self.UNK = "[UNK]"
        self.BOS = "[BOS]"
        self.EOS = "[EOS]"
        self.MASK = "[MASK]"
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.mask_token_id = 4
        self.vocab2idx = {self.PAD: 0, self.UNK: 1, self.BOS: 2, self.EOS: 3, self.MASK: 4}
        self.special_tokens = [self.PAD, self.UNK, self.BOS, self.EOS, self.MASK]

    def update_scores(self):
        total = sum(self.vocab2score.values())
        score_dict = {}
        for key, cnt in self.vocab2score.items():
            if cnt == 0:
                score_dict[key] = math.inf
            else:
                score_dict[key] = -math.log(cnt / total)
        # deal with UNK
        for token in self.special_tokens:
            score_dict[token] = math.inf
        self.vocab2score = score_dict

    def load_pretrained_tokenizer(self, config_path, model_path):
        config = AutoConfig.from_pretrained(config_path)
        model = FastR2D2Plus(config)
        if os.path.exists(os.path.join(model_path, f'model.bin')):
            model.from_pretrain(os.path.join(model_path, f'model.bin'))
        else:
            print("WARNING: no model.bin in pretrain dir!")
        return model

    def load_basic_char_tokenizer(self, model_path, uncased=False):
        from utils.tokenizer import CharLevelTokenizer
        char_tokenizer = CharLevelTokenizer(uncased=uncased)
        char_tokenizer.load_from_vocab_file(os.path.join(model_path, 'vocab.txt'))
        return char_tokenizer

    def find_best_segments(self, root, tokens):
        if self.normalizer:
            tokens = self.normalizer.normalize_str(tokens)

        segments = []
        to_visit = [root]
        while len(to_visit) > 0:
            top = to_visit.pop(-1)
            key = self.continuing_prefix + tokens[top.i : top.j + 1] if top.i != 0 else tokens[top.i : top.j + 1]
            if key in self.vocab2idx:
                segments.append(key)
            # not hit: going deeper
            elif top.left is not None and top.right is not None:
                # will visit left first
                to_visit.append(top.right)
                to_visit.append(top.left)
            else:
                segments.append(self.UNK)
        post_processed_segments = self.post_merge(segments)
        out = []
        if self.continuing_prefix != '':
            for i, seg in enumerate(post_processed_segments):
                if i!=0 and not seg.startswith(self.continuing_prefix):
                    seg = self.continuing_prefix+seg
                out.append(seg)
        else:
            out = post_processed_segments
            
        return out

    def train(self, dataset_path, ext_vocab_path=None, batch_size=32, segtable_size=50000,
              output_dir=None):
        """
        prepare:
            a vocab2idx dict from ext_vocab_path,
            a word2seg dict recording all the parser best segments
        """
        if ext_vocab_path:
            self.load_vocab_file(ext_vocab_path)

        if dataset_path is None:
            return
        
        ## create vocab2seg
        from utils.tree_utils import get_tree_from_merge_trajectory
        from torch.utils.data import DataLoader, SequentialSampler
        from tqdm import tqdm

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device("cuda")
        
        parser = self.model.parser
        parser.to(device)
        parser.eval()

        dataset = WordDataset(dataset_path, self.char_tokenizer, preprocessed=False) 
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SequentialSampler(dataset),
                                collate_fn=dataset.collate_batch,
                                num_workers=1)
        epoch_iterator = tqdm(dataloader, desc="Dataset Iteration")
        
        # inference: create splits
        for step, inputs in enumerate(epoch_iterator):
            with torch.no_grad():
                inputs_device = {}
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs_device[k] = v.to(device)
                    
                with torch.no_grad():
                    s_indices, _ = parser(**inputs_device)
                s_indices = s_indices.cpu().data.numpy()
                tokens = inputs['words']
                seq_lens = inputs['attention_mask'].sum(dim=-1).cpu().data.numpy()
                for sent_i, seq_len in enumerate(seq_lens):
                    try:
                        root = get_tree_from_merge_trajectory(s_indices[sent_i], seq_len)
                    except IndexError as e:
                        print(s_indices[sent_i], seq_len)
                        raise IndexError

                    token = tokens[sent_i]
                    if token not in self.vocab2seg:
                        if len(token) == 1:
                            continue
                        segments = self.find_best_segments(root, token)
                        self.vocab2seg[token] = segments

        if output_dir:
            seg_file = os.path.join(output_dir, "segments.txt")
            vocab_file = os.path.join(output_dir, "vocab.txt")
            with open(seg_file, mode="w") as f_seg:
                for key, segs in self.vocab2seg.items():
                    f_seg.write(key+'\t'+' '.join(segs)+'\n')
            with open(vocab_file, mode="w") as f_vocab:
                for word, idx in self.vocab2idx.items():
                    score = self.vocab2score[word] if idx >= len(self.special_tokens) else 0
                    f_vocab.write(word+'\t'+str(idx)+'\t'+str(score)+'\n')

    def load_vocab_file(self, vocab_file):
        ## create vocab2idx
        with open(vocab_file, 'r') as f_in:
            for l in f_in.readlines():
                word, freq = l.strip().split()
                if word not in self.vocab2idx:
                    self.vocab2idx[word] = len(self.vocab2idx)
                    self.vocab2score[word] = int(freq)
        self.update_scores()
        self.idx2vocab = {idx:key for key,idx in self.vocab2idx.items()}
        self.vocab_size = len(self.idx2vocab)

    def convert_tokens_to_ids(self, x: Union[str, List[str]], BOS=None, EOS=None, UNK="[UNK]", 
                                max_length=None) -> Union[int, List[int]]:

        if isinstance(x, str):
            x_idx = self.vocab2idx[x] if x in self.vocab2idx else self.vocab2idx[UNK]
            return x_idx

        if isinstance(x, List):
            x_idx = [self.vocab2idx[w] if w in self.vocab2idx else self.vocab2idx[UNK] for w in x]
        
            if BOS:
                x_idx = [self.vocab2idx[BOS]] + x_idx
            if EOS:
                x_idx = x_idx + [self.vocab2idx[EOS]]
            if max_length and len(x_idx) > max_length:
                if EOS:
                    x_idx[max_length-1] = self.vocab2idx[EOS]
            x_idx = x_idx[:max_length]
            return x_idx

        else:
            raise TypeError("input should be a str or List[str]!")
    
    def convert_ids_to_tokens(self, x_idx):
        """
            :param x_idx: List[int]
            :return: List[char]
        """
        out = []
        for idx in x_idx:
            x = self.idx2vocab[idx]
            if x == self.EOS:
                break
            if x != self.BOS:
                out.append(x)
        return out
    
    def tokenize(self, word:str) -> List[str]:
        """   
            tokenize word to subwords.
            :param txt: str
            :return: List[str] List of chars

            Taken care of special tokens.

        """
        assert isinstance(word, str), "Input should be a string."

        if word in self.special_tokens:
            tokens = [word]
        else:
            if self.normalizer:
                word = self.normalizer.normalize_str(word)

            if len(word) == 1:
                tokens = [word]
            else:
                tokens = self.vocab2seg.get(word)
                if tokens is None:
                    tokens = self.inference(word)

        return tokens

    def post_merge(self, orig_tokens:List[str]) -> List[str]:
        n = len(orig_tokens)
        if n <= 1:
            tokens = orig_tokens
        else:
            def remove_prefix(subword):
                if subword.startswith(self.continuing_prefix):
                    return subword[2:]
                return subword

            scores = np.full((n, n), np.inf)
            segments = np.empty((n, n), dtype=object)
            # base case init
            for i in range(n):
                scores[i, i] = self.vocab2score[orig_tokens[i]]
                segments[i, i] = [orig_tokens[i]]
            # dp: len 2 -> n
            for h in range(2, n + 1): # len 2 -> n
                for i in range(0, n - h + 1): # st
                    j = i + h - 1 # ed
                    best_k = -1
                    merged_word = ''.join(orig_tokens[i:j+1])
                    # deal with boundary markers
                    if self.continuing_prefix != '':
                        merged_word = orig_tokens[i] + ''.join([remove_prefix(w) for w in orig_tokens[i+1:j+1]])
                    # init best_score as merge score 
                    best_score = self.vocab2score.get(merged_word, np.inf)
                    # iterate all splits
                    for k in range(i, j):
                        if scores[i, k] + scores[k+1, j] <= best_score:
                            best_k = k
                            best_score = scores[i, k] + scores[k+1, j]
                    if best_k != -1: 
                        segments[i, j] = segments[i, best_k] + segments[best_k+1, j] 
                    else: 
                        # merge
                        segments[i, j] = [merged_word]
                    scores[i, j] = best_score
            tokens = segments[0, n-1]
        return tokens

    def encode(self, sent:str, tensor=False) -> List[int]:
        ids = []
        for word in sent.strip().split():
            tokens = self.tokenize(word)
            token_ids = self.convert_tokens_to_ids(tokens)
            ids.extend(token_ids)

        return ids

    def inference(self, word: Union[str, List[str]], log_tree=False) -> List[str]:
        device = torch.device('cpu')
        parser = self.model.parser.to(device)
        parser.eval()

        attention_mask_batch = []
        char_token_ids_batch = []

        char_tokens = self.char_tokenizer.tokenize(word)
        char_token_ids = self.char_tokenizer.convert_tokens_to_ids(char_tokens, tensor=False)
        char_token_ids_batch.append(char_token_ids)
        attention_mask_batch.append([1]*len(char_token_ids))

        attention_mask_batch = torch.tensor(attention_mask_batch)
        char_token_ids_batch = torch.tensor(char_token_ids_batch)

        char_token_ids_batch = char_token_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)

        with torch.no_grad():
            s_indices, _ = parser(input_ids=char_token_ids_batch, attention_mask=attention_mask_batch, noise_coeff=0.0)
        s_indices = s_indices[0].cpu().data.numpy()
        seq_len = len(char_token_ids)
        from utils.tree_utils import get_tree_from_merge_trajectory
        root, tree_str = get_tree_from_merge_trajectory(s_indices, seq_len, char_tokens)
        if log_tree:
            print(
                    f"inputs: {''.join(char_tokens)} "
                    f"parsed tree: {tree_str} \n"
                )
        return self.find_best_segments(root, word)

if __name__ == '__main__':
    import os
    from utils.tree_tokenizer import TreeTokenizer
    tokenizer_path = 'out/1w_singlecomp'

    tokenizer_config_path = os.path.join(tokenizer_path, "all_config.json")
    tokenizer = TreeTokenizer(config_path=tokenizer_config_path, 
                            model_path=tokenizer_path)
    tokenizer.train(None, ext_vocab_path='vocab_mining/1w_singlecomp/vocab_mining_read.out.readable', 
                    batch_size=1024, output_dir=None)
    
    # print(tokenizer.inference("uniquenesses"))
    # print(tokenizer.inference("eyewitnesses"))
    # print(tokenizer.inference("batting"))
    # print(tokenizer.inference("disengage"))
    # print(tokenizer.inference("archeologists"))
    # print(tokenizer.inference("photographers"))
    # print(tokenizer.inference("undesirable"))
    # print(tokenizer.inference("windsurfing"))

    # print(tokenizer.inference("nanotechnology"))
    # print(tokenizer.inference("destabilizing"))
    # print(tokenizer.inference("stabilizing"))
    # print(tokenizer.inference("unlockable"))
    # print(tokenizer.inference("destigmatize"))






            
            
