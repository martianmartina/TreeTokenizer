# coding=utf-8
# Copyright (c) 2021 Ant Group
import os
import random
from bisect import bisect_right, bisect_left
from itertools import accumulate
from torch.utils.data import Dataset
import torch
from typing import List, Dict, Optional, overload
from utils.misc import align_spans, get_sentence_from_words
from utils.vocab_builder import load_span_tokenizer
from abc import ABC, abstractmethod
import linecache
import logging
import pickle
import numpy as np
import codecs

EMPTY_HISTORY = "[EMPTY]"
AGENT = "[AGENT]"
USER = "[USER]"
TOPIC = "[TOPIC]"


logger = logging.getLogger(__name__)


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class InputItem:
    def __init__(self, ids, atom_spans=None, **kwargs) -> None:
        self.ids = ids
        self.atom_spans = atom_spans
        self.kwargs = kwargs

    def __getattr__(self, key):
        if key in self.kwargs:
            return self.kwargs[key]
        else:
            return None

class SamplingByLengthDataset(Dataset, ABC):
    def __init__(self,
                data_path,
                max_seq_len,
                sampling_times=100,
                weighted=True,
                min_len=2,
                max_len=9999,
                max_line=-1,
                cache_dir: Optional[str] = None,
                descending=True,
                sample_across_doc=True,
                **kwargs) -> None:
        super().__init__()
        self._data_path = data_path
        self._max_seq_len = max_seq_len
        self._weighted = weighted
        self._min_len = min_len
        self._max_len = max_len
        self._max_line = max_line
        self._cache_dir = cache_dir
        self._shuffle_id = 0
        self._descending = descending
        self._weighting, self._total_len = None, None
        self._lines = self.load_dataset(self._data_path)
        self._len_dataset = len(self._lines)
        self._num_samples = sampling_times*min(self._len_dataset, self._max_line) if self._max_line>0 else sampling_times*self._len_dataset
        self._sample_across_doc = sample_across_doc
        self.init_weighting()

    def __len__(self):
        return self._num_samples
    
    def load_dataset(self, data_path):
        _lines = []
        with codecs.open(data_path, mode='r', encoding='utf-8') as f_in:
            for line in f_in:
                if len(line.strip().split())>=self._min_len+2:
                    _lines.append(line)
                    if self._max_line > 0 and len(_lines) >= self._max_line:
                        break
        return _lines

    def init_weighting(self):
        if self._weighted:
            lens = np.array([len(l) for l in self._lines])
            self._total_len = np.sum(lens)
            print(f"Dataset line count {len(lens)}, character count {self._total_len}")
            self._weighting = list(accumulate(lens))
        else:
            self._weighting = None

    def get_weighted_samples(self, np_rng):
        if self._weighting is not None:
            idx = np_rng.randint(self._total_len)
            return bisect_right(self._weighting, idx)
        else:
            return np_rng.randint(self._len_dataset)

    def getidx(self, data_idx):
        return self._lines[data_idx]

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
        
        # get possibly weighted random index from dataset
        data_idx = self.get_weighted_samples(rng)
        #        data_idx = rng.choice(self.ds_len, p=self.weighting)
        tokens = self.getidx(data_idx)

        # truncate or pad tokens
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self._max_seq_len
        # print("inside dataset: max seq len", self._max_seq_len)
        # randomly choose a position for start
        if tokens_to_strip > 0:
            strip_left_tokens = rng.randint(tokens_to_strip + 1)
            tokens = tokens[strip_left_tokens:]

            strip_right_rokens = len(tokens) - self._max_seq_len
            if strip_right_rokens > 0:
                tokens = tokens[:-strip_right_rokens]
    
        # Sample multiple documents/sentences
        if self._sample_across_doc:
            while (len(tokens) < self._max_seq_len):
                data_idx = (data_idx + 1) % self._len_dataset
                new_tokens = self.getidx(data_idx)
                tokens = tokens+' '+new_tokens

            tokens = tokens[:self._max_seq_len]
        # print("len tokens", tokens)
        # print("finished", len(tokens))
        return tokens


class SamplingByFreqDataset(Dataset, ABC):
    """
        FIXME: current WordDataset does not support setting max_line
                bc sampling by freq will be infected by incorrect freq.
    """
    def __init__(self,
                data_path_or_dir, 
                tokenizer,
                sampling_times=100,
                weighted=True,
                min_len=2,
                max_len=9999,
                batch_max_len=75000,
                max_line=-1,
                cache_dir: Optional[str] = None,
                descending=True,
                external_vocab_path=None,
                **kwargs) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._weighted = weighted
        self._min_len = min_len
        self._max_len = max_len
        self._batch_max_len = batch_max_len
        self._max_line = max_line
        self._data_path = data_path_or_dir
        self._cache_dir = cache_dir
        self._shuffle_id = 0
        self._descending = descending

        self._lines, self._freq_list = self._load_dataset(data_path_or_dir, **kwargs)
        self._len_dataset = len(self._lines)
        self._num_samples = sampling_times*self._len_dataset

        if external_vocab_path is not None:
            self._span_tokenizer = load_span_tokenizer(external_vocab_path)
        else:
            self._span_tokenizer = None

        self._weighting, self._total_freq = None, None
        self.init_weighting()

    def __len__(self):
        return self._num_samples

    def init_weighting(self):
        if self._weighted:
            self._freq_list = np.array(self._freq_list)
            self._total_freq = np.sum(self._freq_list)
            print("total_freq", self._total_freq)
            self._weighting = list(accumulate(self._freq_list))
        else:
            self._weighting = None

    def get_weighted_samples(self, np_rng):
        if self._weighting is not None:
            idx = np_rng.randint(self._total_freq)
            
            return bisect_right(self._weighting, idx)
        else:
            return np_rng.randint(self._len_dataset)

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        data_idx = self.get_weighted_samples(rng)
        #        data_idx = rng.choice(self.ds_len, p=self.weighting)
        input_item = self._lines[data_idx]
        # make a copy, keep original intact
        tokens = input_item.ids.copy() 
        
        return InputItem(np.array(tokens), atom_spans=None, tokens=input_item.tokens)

    def find_all_hitspans(self, ids):
        """
        Given a sentence, return all the spans (st, ed, span_id) that hit an external vocab .

        :param ids: a list of token ids in a tokenized sentence.
        :return: a 3-ary list 
                3k-th elems: span start idx (inclusive)
                3k+1-th elems: span end idx (inclusive)
                3k+2-th elems: span vocab idx (original + 1)
        """
        results = self._span_tokenizer.tokenize(ids)
        span_idx = np.zeros((len(results),), dtype=np.int32)
        if len(results) > 0:
            assert len(results) % 3 == 0
            for group_id in range(len(results) // 3):
                idx, span_len, span_id = results[group_id * 3: group_id * 3 + 3]
                assert span_id >= 0
                span_idx[group_id * 3] = idx - span_len + 1
                span_idx[group_id * 3 + 1] = idx
                ## NOTE: 0 is reserved for spans that miss the ext_vocab
                span_idx[group_id * 3 + 2] = span_id
        return span_idx
        
    @abstractmethod
    def _load_dataset(self, data_path_or_dir, **kwargs) -> List[InputItem]:
        pass

class WordSamplingDataset(SamplingByFreqDataset):
    """
        FIXME: current WordDataset does not support setting max_line
                bc sampling by freq will be infected by incorrect freq.
    """
    def _load_dataset(self, data_path_or_dir, **kwargs) -> List[InputItem]:
        print("loading dataset.")

        input_item_list = []
        freq_list = []
        lines = linecache.getlines(data_path_or_dir)
       
        for _line in lines:
            ori, freq = _line.strip().split()
            tokens = self._tokenizer.tokenize(ori)
            token_ids = self._tokenizer.convert_tokens_to_ids(tokens)

            if self._min_len < len(token_ids) < self._max_len:
                input_item_list.append(InputItem(np.array(token_ids), atom_spans=None, tokens=tokens))
                freq_list.append(int(freq))

        print("Loading finished.")
        return input_item_list, freq_list
    
    def collate_batch(self, items: List[InputItem]) -> Dict[str, torch.Tensor]: 
        batch_len = 0 ## track num_tokens within this batch
        ids_batch = []
        tokens_batch = []
        ## NOTE: cut at batch_max_token_num
        for item in items:
            if batch_len + len(item.ids) <= self._batch_max_len:
                batch_len += len(item.ids)
                ids_batch.append(item.ids)
                tokens_batch.append(item.tokens)
            else: 
                break

        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))
        list_lens = [len(x) for x in ids_batch]

        input_ids_batch = []
        mask_batch = []
        span_indices = []

        for input_ids in ids_batch:
            padded_input_ids = np.append(np.array(input_ids), np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_input_ids)
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
            if self._span_tokenizer is not None:
                span_idx = self.find_all_hitspans(input_ids) # (3*span_num, )
                span_indices.append(span_idx)

        model_inputs = {
            "input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
            "input_tokens": tokens_batch,
            "masks": torch.tensor(np.array(mask_batch), dtype=torch.long),
            "seq_len":list_lens,
            "group_ids": np.arange(len(ids_batch)),
            "span_ids": span_indices
        }
        parser_inputs = {
            "input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
            "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long),
        }

        return model_inputs, parser_inputs

class MorphSegWordDataset(SamplingByFreqDataset):
    def _load_dataset(self, data_path_or_dir, **kwargs) -> List[InputItem]:
        print("loading dataset.")

        input_item_list = []
        freq_list = []
        lines = linecache.getlines(data_path_or_dir)
       
        for _line in lines:
            ori, seg, freq = _line.strip().split()
            tokens = self._tokenizer.tokenize(seg)
            token_ids = self._tokenizer.convert_tokens_to_ids(tokens)

            if self._min_len < len(token_ids) < self._max_len:
                input_item_list.append(InputItem(np.array(token_ids), atom_spans=None, ori=ori))
                freq_list.append(int(freq))

        print("Loading finished.")
        return input_item_list, freq_list
    
    def collate_batch(self, items: List[InputItem]) -> Dict[str, torch.Tensor]: 
        ids_batch = [item.ids for item in items]
        ori_batch = [item.ori for item in items]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        input_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            padded_input_ids = np.append(np.array(input_ids), np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_input_ids)
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))

        model_inputs = {
            "input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
            "masks": torch.tensor(np.array(mask_batch), dtype=torch.long),
            "ori": ori_batch
        }
        parser_inputs = {
            "input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
            "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long),
        }

        return model_inputs, parser_inputs