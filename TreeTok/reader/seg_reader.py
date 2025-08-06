import json
import codecs
import numpy as np
import torch
import os
from reader.memory_line_reader import InputItem
from transformers import PreTrainedTokenizerFast
from utils.tokenizer import CharLevelTokenizer
from abc import ABC
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, DistributedSampler
from typing import List, Dict

def process_spans(spans):
    """
        ['abs', 'tract'] -> [(0,2), (3,7)]
    """
    ret = []
    st = 0
    for span in spans:
        ret.append((st, st+len(span)-1))
        st += len(span)
    return ret
    
class MorphSegReader(Dataset, ABC):
    """
    For the morphological segmentation eval dataset
    """

    def __init__(self, path, tokenizer,
                 min_len=2, max_line=-1,
                 uncase=False, output_path=None,
                 random=False, **kwargs):
        '''
        params:
        random: True: for randomly batch sentences
                False: batch sentences in similar length
        '''
        super().__init__()
        self._path = path
        self._tokenizer = tokenizer
        self._min_len = min_len
        self._max_line = max_line
        self.random = random
        self._uncase = uncase
        self._output_path = output_path
        self._lines = self._load_dataset(self._path, output_path=output_path)

    def _load_dataset(self, data_path_or_dir, output_path=None, **kwargs) -> List[InputItem]:
        input_item_list = []
        with open(data_path_or_dir, 'r') as f:
            for l in f.readlines():
                # raw input lines are like:
                # ahead a head [, a head]*
                _ = l.strip().split(",")

                # inputs: List[List[char]]
                orig = _[0].split()[0]
                seg1 = ' '.join(_[0].split()[1:])
                tokens = self._tokenizer.tokenize(orig)
                token_ids = self._tokenizer.convert_tokens_to_ids(tokens)

                # labels: List[List[str]]
                labels = []
                segs = [seg1] + _[1:]

                for seg in segs:
                    spans = seg.split()
                    span_ids = process_spans(spans)
                    labels.append(span_ids)

                input_item_list.append(InputItem(token_ids, ori_word=orig, labels=labels, label_tokens=segs))
                if len(input_item_list) > self._max_line > 0:
                    break
        print(f"Total number of examples {len(input_item_list)}")
        if output_path is not None:
            with open(output_path, 'w') as f:
                for item in input_item_list:
                    f.write(item.ori_word+'\n')
        return input_item_list

    def __getitem__(self, idx):
        return self._lines[idx]

    def __len__(self):
        return len(self._lines)

    def collate_batch(self, items: List[InputItem]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items]
        labels_batch = [item.kwargs['labels'] for item in items]
        tokens_batch = [item.kwargs['label_tokens'] for item in items]
        ori_words_batch = [item.kwargs['ori_word'] for item in items]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))
        input_ids_batch = []
        mask_batch = []
        input_labels_batch = []

        for input_ids, label_ids in zip(ids_batch, labels_batch):
            input_ids_batch.append(input_ids + [self._tokenizer.pad_token_id] * (input_max_len - len(input_ids)))
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
            input_labels_batch.append(label_ids)
        
        model_inputs = {"input_ids": torch.tensor(input_ids_batch),
                        "masks": torch.tensor(mask_batch),
                        "group_ids": np.arange(len(items)),
                        "ori_word": ori_words_batch,
                        "labels":input_labels_batch,
                        "label_tokens":tokens_batch}
        parser_inputs = {"input_ids": torch.tensor(input_ids_batch),
                        "attention_mask": torch.tensor(mask_batch)
                        }

        return model_inputs, parser_inputs

class CSVReader(MorphSegReader, ABC):
    def __init__(self, path, tokenizer, lang, uncase=False, output_path=None, **kwargs):
        self._lang = lang
        self._uncase = uncase
        super().__init__(path, tokenizer)
        self._tokenizer = tokenizer
        self._lines = self._load_dataset(path, output_path=output_path)

    def _load_dataset(self, data_path_or_dir, output_path=None, **kwargs) -> List[InputItem]:
        input_item_list = []
        cnt = 0
        import pandas as pd
        data = pd.read_csv(data_path_or_dir)
        ## only load English compound words
        filtered_data = data[(data['lang'] == self._lang) & (data['type'] == 'positive')]
        X_train = filtered_data[['word', 'segmentation']]
        for row in X_train.itertuples():
            if len(row.segmentation.split('-')) < 2:
                continue
            else:
                cnt += 1
                word = row.word
                segmentation = row.segmentation
            if self._uncase:
                word = word.lower()
                segmentation = segmentation.lower()
            tokens = self._tokenizer.tokenize(word)
            if isinstance(self._tokenizer, PreTrainedTokenizerFast):
                token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
            else:
                token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
            span_bounds = process_spans(segmentation.split('-'))
            gold_segment = segmentation.replace('-', ' ')
            input_item_list.append(InputItem(token_ids, ori_word=word, labels=[span_bounds], label_tokens=[gold_segment]))
        if output_path is not None:
            with open(output_path, 'w') as f:
                for item in input_item_list:
                    f.write(item.ori_word+'\n')
        print(f"Total number of examples {cnt}")
        return input_item_list
    
    def __getitem__(self, idx):
        return self._lines[idx]

    def __len__(self):
        return len(self._lines)
    
if __name__ == '__main__':
    from tqdm import tqdm
    preload_vocab_path = 'data/wiki103/wiki_basic_vocab.txt'
    corpus_path = 'data/compound/valid.csv'
    tokenizer = CharLevelTokenizer()
    tokenizer.load_from_vocab_file(preload_vocab_path)
    dataset = CSVReader(corpus_path, tokenizer)
    sampler = SequentialSampler(dataset)
    collate_fn = dataset.collate_batch
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            sampler=sampler,
                            collate_fn=collate_fn)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, inputs_pair in enumerate(epoch_iterator):
        if step>5:
            break
        print(inputs_pair)