from typing import List, Dict, Any
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

class WordDataset(Dataset):
    """
    A dataset to handle word frequencies.

    It can load data from two formats, controlled by the `preprocessed` flag:
    1. A raw text corpus (preprocessed=False).
    2. A pre-calculated file where each line is "word frequency" (preprocessed=True).
    """
    def __init__(self, corpus_path: str, tokenizer: Any, preprocessed: bool = False):
        self._tokenizer = tokenizer
        self.preprocessed = preprocessed
        self._dataset = self._load_dataset(corpus_path)
        self._ds_len = len(self._dataset)
        print(f"Finished loading the dataset. Found {self._ds_len} unique words.")

    def _load_dataset(self, corpus_path: str) -> List[List]:
        data = []
        if self.preprocessed:
            print("Loading from a preprocessed word-frequency file...")
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2: continue
                    word, freq_str = parts

                    if not (1 <= len(word) <= 250): continue
                    
                    tokens = self._tokenizer.tokenize(word)
                    ids = self._tokenizer.convert_tokens_to_ids(tokens)
                    data.append([ids, word, int(freq_str)])
        else:
            print("Building word frequencies from raw corpus...")
            word_freq_map = defaultdict(int)
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    for word in line.strip().split():
                        word_freq_map[word] += 1
            
            for word, freq in word_freq_map.items():
                tokens = self._tokenizer.tokenize(word)
                ids = self._tokenizer.convert_tokens_to_ids(tokens)
                data.append([ids, word, freq])
        
        return data

    def __len__(self) -> int:
        return self._ds_len

    def __getitem__(self, idx: int) -> List:
        return self._dataset[idx]
    
    def collate_batch(self, batch: List[List]) -> Dict[str, Any]:
        all_ids, all_words, all_freqs = zip(*batch)
        
        max_len = max(len(ids) for ids in all_ids)
        
        padded_ids = np.zeros((len(batch), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)

        for i, ids in enumerate(all_ids):
            length = len(ids)
            padded_ids[i, :length] = ids
            attention_mask[i, :length] = 1
        
        return {
            "input_ids": torch.from_numpy(padded_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "words": all_words,
            "freqs": torch.tensor(all_freqs, dtype=torch.long)
        }