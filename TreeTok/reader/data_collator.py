from typing import List, Dict
import math
import torch
import numpy as np
from reader.memory_line_reader import InputItem
from multiprocessing.pool import ThreadPool
from collections import OrderedDict
from utils.vocab_builder import load_span_tokenizer
import codecs

class CharacterCollator:
    def __init__(self, tokenizer, mlm_rate=0.15, mlm_replace_rate=0.2, external_vocab_path=None, suffix_offset=0):
        self._tokenizer = tokenizer
        self._mlm_rate = mlm_rate
        self._mlm_replace_rate = mlm_replace_rate # prob of randomly replacing with another token
        if external_vocab_path is not None:
            self._span_tokenizer = load_span_tokenizer(external_vocab_path)
            self._suffix_offset = suffix_offset
        else:
            self._span_tokenizer = None
            self._suffix_offset = 0

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
                st = idx - span_len + 1
                span_idx[group_id * 3] = st
                span_idx[group_id * 3 + 1] = idx
                if st == 0:
                    span_idx[group_id * 3 + 2] = span_id
                else:
                    span_idx[group_id * 3 + 2] = span_id + self._suffix_offset
        return span_idx

    def eval_word_collate_fn(self, items: List[InputItem]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items]
        labels_batch = [item.kwargs['labels'] for item in items]
        tokens_batch = [item.kwargs['label_tokens'] for item in items]
        ori_words_batch = [item.kwargs['ori_word'] for item in items]
        lens = map(lambda a: len(a), ids_batch)
        lens_list = [len(a) for a in ids_batch]
        input_max_len = max(1, max(lens))
        input_ids_batch = []
        mask_batch = []
        input_labels_batch = []
        span_indices = []

        for input_ids, label_ids in zip(ids_batch, labels_batch):
            input_ids_batch.append(input_ids + [self._tokenizer.pad_token_id] * (input_max_len - len(input_ids)))
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
            input_labels_batch.append(label_ids)
            if self._span_tokenizer is not None:
                span_indices.append(self.find_all_hitspans(input_ids)) # (3*span_num, )
        
        model_inputs = {"input_ids": torch.tensor(input_ids_batch),
                        "masks": torch.tensor(mask_batch),
                        "group_ids": np.arange(len(items)),
                        "span_ids": span_indices,
                        "ori_word": ori_words_batch,
                        "labels":input_labels_batch,
                        "label_tokens":tokens_batch,
                        "seq_len":lens_list}
        parser_inputs = {"input_ids": torch.tensor(input_ids_batch),
                        "attention_mask": torch.tensor(mask_batch)
                        }

        return model_inputs, parser_inputs

    def lm_collate_fn(self, str_list) -> Dict[str, torch.Tensor]:
        empty_str_cnt = 0
        """
        # tokens: [['h','o','w'], ['i','s'],['l','i','f','e']]
        # input_ids = [[0,1,2],[3,4],[5,3,6,7]]
        # chunk_ids = [0,1,2,3,4,5,3,6,7]
        # # NOTE: chunk_mask = [1,1,1,2,2,3,3,3,3] (padding is set to 0) NOTE
        ## previously chunk mask padding is -1
        """
        input_ids_list = []
        chunk_input_ids_list = []
        chunk_mask_list = []
        span_indices = []
        group_ids = []
        chunk_word_num_list = []

        # things that need padding
        input_ids_batch = []
        chunk_input_ids_batch = []
        chunk_mask_batch = []
        mask_batch = []
        # word-level: for gpt inputs
        padded_group_ids = [] # map from flatten to 2d
        gpt_gather_ids = [] # map from 2d to flatten
        gpt_mask_batch = [] 
        tokens_batch = []
        ntp_list = []  # next token predict list

        word2idx = {}
        inbatch_root_ids = []
        w_idx = 0
        w_cnt = 0
        
        for group_id, s in enumerate(str_list):
            next_token_prediction = []
            ## NOTE: s is a single chunk
            # in character-level LM
            # valid_tokens ensures that we only process whole words
            valid_token_list = s.split()[1:-1]
            if len(valid_token_list) == 0:
                valid_token_list = ['[UNK]']
                
            for w in valid_token_list:
                if w not in word2idx:
                    word2idx[w] = w_idx
                    inbatch_root_ids.append(w_cnt)
                    w_idx += 1
                w_cnt += 1
            for w in valid_token_list[1:]:
                next_token_prediction.append(word2idx[w])
            ntp_list.append(next_token_prediction)
            tokens_batch.append(valid_token_list)
            tokens = [[c for c in valid_token] for valid_token in valid_token_list] # List[List[str]]
            # prepare r2d2 model inputs
            input_ids_list_per_chunk = [self._tokenizer.convert_tokens_to_ids(word) for word in tokens]
            input_ids_list.extend(input_ids_list_per_chunk)
            group_ids.extend([group_id]*len(tokens))
            chunk_word_num_list.append(len(tokens))
            # prepare parser inputs
            chunk_input_ids = [word_id for word in input_ids_list_per_chunk for word_id in word]
            chunk_input_ids_list.append(chunk_input_ids)
            lengths = [len(word) for word in tokens]
            chunk_mask = np.repeat(1+np.arange(len(tokens)), lengths) # chunk attn_mask starts from 1
            chunk_mask_list.append(chunk_mask)

            if self._span_tokenizer is not None:
                span_indices.extend([self.find_all_hitspans(input_ids) for input_ids in input_ids_list_per_chunk]) # (3*span_num, )
            else:
                span_indices.extend([np.array([]) for input_ids in input_ids_list_per_chunk])
                
        lens = map(lambda a: len(a), input_ids_list)
        input_max_len = max(1, max(lens))

        chunk_lens = map(lambda a: len(a), chunk_input_ids_list)
        chunk_input_max_len = max(1, max(chunk_lens))

        max_chunk_word_num = max(chunk_word_num_list)
        
        # make paddings and masks
        for input_ids in input_ids_list:
            padded_input_ids = np.append(np.array(input_ids), np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_input_ids)
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
        for i, chunk_input_ids in enumerate(chunk_input_ids_list):
            padded_chunk_input_ids = np.append(np.array(chunk_input_ids), np.array([self._tokenizer.pad_token_id] * (chunk_input_max_len - len(chunk_input_ids))))
            chunk_input_ids_batch.append(padded_chunk_input_ids)
            chunk_mask = chunk_mask_list[i]
            chunk_mask_padding = 0*np.ones(chunk_input_max_len - len(chunk_mask), dtype=int) # chunk_mask pad 0
            padded_chunk_mask = np.concatenate((chunk_mask, chunk_mask_padding), axis=0)
            chunk_mask_batch.append(padded_chunk_mask)
        # for gpt
        idx = 0
        for batch_id, chunk_word_num in enumerate(chunk_word_num_list):
            gpt_gather_ids.extend(np.arange(chunk_word_num)+idx)
            idx += max_chunk_word_num # iter to next row
            assert len(ntp_list[batch_id]) <= max_chunk_word_num, (len(ntp_list[batch_id]), max_chunk_word_num)
            ntp_list[batch_id] = ntp_list[batch_id] + [-1] * (max_chunk_word_num - len(ntp_list[batch_id]))
        
        idx = 0
        for chunk_word_num in chunk_word_num_list:
            paddings = np.full(max_chunk_word_num-chunk_word_num, -1)
            padded_group_ids.append(np.concatenate((np.arange(chunk_word_num)+idx, paddings)))
            idx += chunk_word_num
        padded_group_ids = np.array(padded_group_ids)
        gpt_mask_batch = np.where(padded_group_ids == -1, 0, 1)
        
        model_inputs = {
            "input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
            "masks": torch.tensor(np.array(mask_batch), dtype=torch.long),
            "group_ids": np.array(group_ids),
            "padded_group_ids": padded_group_ids, # np.array
            "gpt_masks": torch.tensor(gpt_mask_batch),
            "gpt_gather_ids": np.array(gpt_gather_ids),
            "span_ids": span_indices,
            "tokens": tokens_batch,
            "gpt_tgt": torch.tensor(np.array(ntp_list)),
            "inbatch_sampling": torch.tensor(np.array(inbatch_root_ids))
        }
        parser_inputs = {
            "input_ids": torch.tensor(np.array(chunk_input_ids_batch), dtype=torch.long), 
            "attention_mask": torch.tensor(np.array(chunk_mask_batch), dtype=torch.long)
        }
        return model_inputs, parser_inputs
   
    def mlm_collate_fn(self, str_list):
        """
        tokenization happens here
        """
        input_ids_list = []
        atom_spans_list = []

        input_ids_batch = []
        mask_batch = []
        for s in str_list:
            # valid_tokens ensures that we only process whole words
            valid_tokens = s.split()[1:-1]
            tokens, atom_spans = self._tokenizer.tokenize(valid_tokens, get_atom_spans=True)
            assert len(tokens) > 1, f"having an input{valid_tokens} with tokens less than two"
            input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
            input_ids_list.append(input_ids)
            atom_spans_list.append(atom_spans)

        if input_ids_list:
            lens = map(lambda a: len(a), input_ids_list)
            input_max_len = max(1, max(lens))
        else:
            input_max_len = 1

        org_input_ids_batch = []
        input_ids_batch = []
        target_ids_batch = []
        mask_batch = []

        for input_ids in input_ids_list:
            masked_input_ids = np.array(input_ids)
            target_ids = np.array([-1] * len(masked_input_ids))
            rand_vals = np.random.rand(len(input_ids))
            masked_pos = list(filter(lambda x: rand_vals[x] < self._mlm_rate, range(len(input_ids))))
            for idx in masked_pos:
                target_ids[idx] = input_ids[idx]
                if np.random.rand() < self._mlm_replace_rate:
                    masked_input_ids[idx] = np.random.randint(0, high=self._tokenizer.vocab_size)
                else:
                    masked_input_ids[idx] = self._tokenizer.mask_token_id

            padded_ids = np.append(masked_input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_ids)
            padded_org_ids = np.append(input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            org_input_ids_batch.append(padded_org_ids)
            padded_tgt_ids = np.append(target_ids, np.array([-1] * (input_max_len - len(input_ids))))
            target_ids_batch.append(padded_tgt_ids)
            mask_batch.append(np.array([1] * len(input_ids) + [0] * (input_max_len - len(input_ids))))

        return {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                "tgt_ids": torch.tensor(np.array(target_ids_batch), dtype=torch.long),
                "masks": torch.tensor(np.array(mask_batch), dtype=torch.long),
                "atom_spans": atom_spans_list,
                "pairwise":False},\
               {"input_ids": torch.tensor(np.array(org_input_ids_batch), dtype=torch.long),
                "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long),
                "atom_spans": atom_spans_list}


    def span_mlm_collate_fn(self, str_list):
        """
        tokenization happens here .
        mlm: only masking, no replacing .
        """
        input_ids_list = []
        atom_spans_list = []

        input_ids_batch = []
        mask_batch = []
        for s in str_list:
            # valid_tokens ensures that we only process whole words
            valid_tokens = s.split()[1:-1]
            tokens, atom_spans = self._tokenizer.tokenize(valid_tokens, get_atom_spans=True)
            assert len(tokens) > 1, f"having an input{valid_tokens} with less than two tokens"
            input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
            input_ids_list.append(input_ids)
            atom_spans_list.append(atom_spans)

        lens = map(lambda a: len(a), input_ids_list)
        input_max_len = max(1, max(lens))

        org_input_ids_batch = []
        input_ids_batch = []
        target_ids_batch = []
        mask_batch = []

        for input_ids in input_ids_list:
            masked_input_ids = np.array(input_ids)
            target_ids = np.array([-1] * len(masked_input_ids))
            num_tokens_to_mask = math.ceil(len(input_ids) * self._mlm_rate)
            num_masked_tokens = 0
            
            while num_masked_tokens < num_tokens_to_mask:
                # Sample span length from a geometric distribution and cap at 10 tokens
                span_length = min(np.random.geometric(0.2), 10)
                # Choose starting point for the span
                span_start = np.random.randint(0, len(input_ids) - span_length + 1)
                span_end = span_start + span_length
                for idx in range(span_start, span_end):
                    if num_masked_tokens >= num_tokens_to_mask:
                        break
                    if target_ids[idx] == -1:  # Token isn't already masked
                        target_ids[idx] = input_ids[idx]
                        num_masked_tokens += 1
                        masked_input_ids[idx] = self._tokenizer.mask_token_id
            # Padding
            padded_ids = np.append(masked_input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            input_ids_batch.append(padded_ids)
            padded_org_ids = np.append(input_ids, np.array([self._tokenizer.pad_token_id] * (input_max_len - len(input_ids))))
            org_input_ids_batch.append(padded_org_ids)
            padded_tgt_ids = np.append(target_ids, np.array([-1] * (input_max_len - len(input_ids))))
            target_ids_batch.append(padded_tgt_ids)
            mask_batch.append(np.array([1] * len(input_ids) + [0] * (input_max_len - len(input_ids))))

        return {"input_ids": torch.tensor(np.array(input_ids_batch), dtype=torch.long), 
                "tgt_ids": torch.tensor(np.array(target_ids_batch), dtype=torch.long),
                "masks": torch.tensor(np.array(mask_batch), dtype=torch.long),
                "atom_spans": atom_spans_list,
                "pairwise":False},\
               {"input_ids": torch.tensor(np.array(org_input_ids_batch), dtype=torch.long),
                "attention_mask": torch.tensor(np.array(mask_batch), dtype=torch.long),
                "atom_spans": atom_spans_list}



if __name__ == "__main__":
    from tqdm import tqdm
    from utils.tokenizer import CharLevelTokenizer
    from reader.gpt2_reader import SamplingByLengthDataset
    from reader.data_collator import CharacterCollator
    data_path = "data/wiki103/wiki.debug.tokens"
    preload_vocab_path = "data/wiki103/wiki_basic_vocab.txt"
    ext_vocab_path = "data/wiki103/ext_vocab_bpe.txt.ids"
    tokenizer = CharLevelTokenizer()
    tokenizer.load_from_vocab_file(preload_vocab_path)
    dataset = SamplingByLengthDataset(data_path, max_seq_len=10, max_line=50)
    collator = CharacterCollator(tokenizer, external_vocab_path=ext_vocab_path, external_vocab_size=45679)
    collator_fn = collator.lm_collate_fn
    from torch.utils.data import DataLoader, SequentialSampler
    dataloader = DataLoader(dataset, batch_size=3, sampler=SequentialSampler(dataset),
                            collate_fn=collator_fn, num_workers=1)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, inputs in enumerate(epoch_iterator):
        if step >= 1: break
        print(inputs[0])
        print(inputs[1])
