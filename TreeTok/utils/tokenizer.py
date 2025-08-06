import numpy as np
import torch
from functools import reduce
from typing import List, Dict
from transformers import AutoTokenizer

def load_normalizer(name: str):
    if name == 'bert-base-uncased':
        tokenizer = AutoTokenizer.from_pretrained(name)
        return tokenizer.backend_tokenizer.normalizer
    else:
        raise NotImplementedError    

class CharLevelTokenizer(object):
    def __init__(self, uncased=False):
        self.vocab2idx = dict()
        self.idx2vocab = dict()
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
        self.normalizer = None
        if uncased:
            self.normalizer = load_normalizer('bert-base-uncased')


    def add_tokens(self, new_tokens, special_tokens=False):
        add_count = 0
        if isinstance(new_tokens, list):
            for token in new_tokens:
                if token not in self.vocab2idx:
                    curr_idx = len(self.vocab2idx)
                    self.vocab2idx[token] = curr_idx
                    self.idx2vocab[curr_idx] = token 
                    if special_tokens:
                        self.special_tokens.append(token)
                    add_count += 1
        else:
            if new_tokens not in self.vocab2idx:
                curr_idx = len(self.vocab2idx)
                self.vocab2idx[new_tokens] = curr_idx
                self.idx2vocab[curr_idx] = new_tokens
                if special_tokens:
                    self.special_tokens.append(new_tokens)
                add_count += 1

        return add_count

    def train(self, dataset_path, output_path=None):
        f = open(dataset_path, mode='r')
        for line in f.readlines():
            line = line.strip()
            if self.normalizer:
                line = line.lower() # self.normalizer.normalize_str(line)

            words = line.split()
            for word in words:
                for c in word:
                    if c not in self.vocab2idx:
                        self.vocab2idx[c] = len(self.vocab2idx)
        self.idx2vocab = {idx:key for key,idx in self.vocab2idx.items()}
        self.vocab_size = len(self.idx2vocab)
        if output_path:
            f_out = open(output_path, mode="w")
            for key in self.vocab2idx:
                f_out.write(key+'\n')
    
    def load_from_vocab_file(self, vocab_path, output_path=None, is_master=True):
        # used only in eval_gold_segments.py
        # TODO: support different special characters
        f = open(vocab_path, mode='r')
        for _,line in enumerate(f.readlines()):
            c = line.strip()
            if c not in self.vocab2idx:
                self.vocab2idx[c] = len(self.vocab2idx)
        self.idx2vocab = {idx:key for key,idx in self.vocab2idx.items()}
        self.vocab_size = len(self.idx2vocab)
        print("char tokenizer: vocab size is", len(self.idx2vocab))
        if output_path and is_master:
            f_out = open(output_path, mode="w")
            for key in self.vocab2idx:
                f_out.write(key+'\n')
            f_out.close()
                
    def convert_tokens_to_ids(self, x, BOS=None, EOS=None, UNK="[UNK]", max_length=None, tensor=False):
        """
        :param x: List[str] list of tokens
        :return: List[int] list of ids
        """
        x_idx = [self.vocab2idx[w] if w in self.vocab2idx else self.vocab2idx[UNK] for w in x]
        if BOS:
            x_idx = [self.vocab2idx[BOS]] + x_idx
        if EOS:
            x_idx = x_idx + [self.vocab2idx[EOS]]
        if max_length and len(x_idx) > max_length:
            if EOS:
                x_idx[max_length-1] = self.vocab2idx[EOS]
        x_idx = x_idx[:max_length]
        return torch.Tensor(x_idx).long() if tensor else x_idx
    
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
    
    def tokenize(self, txt):
        """   
            tokenize str to chars.
            :param txt: str
            :return: List[str] List of chars

            Taken care of special tokens.

        """
        assert isinstance(txt, str), "Input should be a string."
        txt = txt.strip()
        if txt in self.special_tokens:
            tokens = [txt]
        else:
            if self.normalizer:
                txt = txt.lower() # self.normalizer.normalize_str(txt)
            tokens = [c for c in txt]
        return tokens
    
    def encode(self, sent:str) -> List[int]:
        ids = []
        for word in sent.strip().split():
            tokens = self.tokenize(word)
            token_ids = self.convert_tokens_to_ids(tokens)
            ids.extend(token_ids)

        return ids
    

class MorphCharLevelTokenizer(CharLevelTokenizer):
    def __init__(self, vocab_path=None):
        super().__init__()
        if vocab_path:
            morph_vocab = open(vocab_path)
            for l in morph_vocab.readlines():
                entry = l.strip().split()[0]
                if entry not in self.vocab2idx:
                    self.vocab2idx[entry] = len(self.vocab2idx)

    def tokenize(self, txt):
        pre_tokens = txt.split("_")
        tokens = [[tok] if tok in self.vocab2idx else [c for c in tok] for tok in pre_tokens]
        tokens = reduce(lambda a,b: a+b, tokens)
        return tokens