# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu
import time
from collections import namedtuple
import copy
import logging
from typing import List, Tuple
from data_structure.py_backend import CPPChartTableManager
from model.r2d2_base import R2D2Base
import torch.nn as nn
import torch
import torch.nn.functional as F
from data_structure.tensor_cache import TensorCache, CacheType
from model.r2d2_common import SPECIAL_TOKEN_NUM, INF_LOG_P_ID
from model.tree_encoder import InsideEncoder, OutsideEncoder, PoolingLayer
from model.fast_parser import TransformerParser


logger = logging.getLogger(__name__)


InsideGroup = namedtuple("InsideGroup", ["parent_ids", "candidate_e_ij_ids", "candidate_log_p_ids", 
                                         "span_lens"])


POOLING_MODE = ["per_layer", "final_layer", "no_pooling"]

HEIGHT_PURNISHMENT = 4

class FastR2D2Plus(R2D2Base):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.inside_only = getattr(config, 'inside_only', False)
        self.inside_enc = InsideEncoder(config)
        
        self.outside_enc = OutsideEncoder(config)
        self.outside_root_embedding = nn.Parameter(torch.rand(config.embedding_dim))

        self.gpt = None
        self.parser = TransformerParser(config)
        
        self.e_ij_id = -1
        self.score_sum_id = -1
        self.score_ijk = -1
        self.height_ij = -1
    
    def initialize_embeddings(self, input_ids, seq_lens):
        # Initialize embeddings
        block_size = input_ids.shape[-1]
        indices_gather = []
        for seq_i, seq_len in enumerate(seq_lens):
            indices_gather.extend(
                range(block_size * seq_i, block_size * seq_i + seq_len))
        if input_ids is not None:
            flatten_input_ids = input_ids.flatten()
            flatten_input_ids = flatten_input_ids.gather(
                dim=0, index=torch.tensor(indices_gather, device=self.device))
            embeddings = self.embedding(flatten_input_ids)
            
        return flatten_input_ids, embeddings
    
    def create_tensor_cache(self, seq_lens, total_cache_size=-1):
        # e_ij, log_p_ij, log_p_sum_ij
        tensor_cache = TensorCache(
            self.window_size,
            seq_lens,
            cache_types=[
                CacheType.NORMAL, CacheType.DETACH,
                CacheType.NORMAL, CacheType.NORMAL
            ],
            dims=[self.input_dim, 1, 1, 1],
            placeholder_num=SPECIAL_TOKEN_NUM,
            device=self.device,
            total_cache_size=total_cache_size)
        self.e_ij_id = 0
        self.score_sum_id = 1
        self.score_ijk = 2
        self.height_ij = 3
        tensor_cache.fill(0, tensor_cache.capacity, [self.height_ij], [0])
        return tensor_cache

    def prepare_composition(self, group_ids, log_p_ids, tensor_cache):
        e_ij, h_ij = tensor_cache.gather(group_ids.flatten(), [self.e_ij_id, self.height_ij])
        log_p_ij = tensor_cache.gather(log_p_ids.flatten(), [self.score_sum_id])[0]
        e_ij = e_ij.view(*group_ids.shape, self.input_dim)
        h_ij = h_ij.view(*group_ids.shape)  # (batch_size, group_size, 2)
        log_p_ij = log_p_ij.view(*group_ids.shape) # (batch_size, group_size, 2)

        return e_ij, log_p_ij.sum(dim=-1), h_ij
    
    def _create_candidate_log_p_pool(self, capacity):
        candidates_log_p_pool = torch.full([capacity, self.window_size],
                                           -1e7, device=self.device)
        return candidates_log_p_pool
    
    def _expand_log_p_ijk_mean(self, log_p_ijk_mean):
        if log_p_ijk_mean.shape[1] < self.window_size:
            # filling with huge neg value
            log_p_ijk_mean_exp = torch.full([log_p_ijk_mean.shape[0], self.window_size], -1e20, device=self.device)
            log_p_ijk_mean_exp[:, :log_p_ijk_mean.shape[1]] = log_p_ijk_mean
            return log_p_ijk_mean_exp
        return log_p_ijk_mean

    def inside(self,
               inside_cache,
               # span_embeds, ## NOTE: self.embedding includes ext_embedding
               inside_groups,
               non_blocking=True):
        splits_orders = []
        split_probabilities = []
        
        for target_cache_ids, span_ids, cache_ids, detach_cache_ids in inside_groups:
            ## NOTE: all `batch_size` indicated here refer to 'num of spans'
            ## per for-loop step, that specifically refer to `num of spans` in a layer*
            '''
                target_cache_ids: (?) #parent node
                span_ids: (?) #pair with target_cache_ids
                cache_ids: (?, group_size, 2) #child nodes
                detach_cache_ids: (?, group_size, 2) #child probs
            '''

            # # if candidate e_ij and log_p is not empty, apply composition function
            e_ij, scores_ij_sum, h_ij = self.prepare_composition(
                cache_ids, detach_cache_ids, inside_cache)
            # # e_ij: (batch_size, group_size, 2, dim), c_ij: (batch_size, 2, dim)

            if self.ext_vocab_size > 0:
                default_span_ids = torch.zeros_like(span_ids)
                if self.no_span_emb:
                    scores_ijk, c_ijk = self.inside_enc(e_ij, self.embedding(default_span_ids))
                else:
                    scores_ijk, c_ijk = self.inside_enc(e_ij, self.embedding(span_ids))
            else:
                default_span_ids = torch.zeros_like(span_ids)
                scores_ijk, c_ijk = self.inside_enc(e_ij, self.embedding(default_span_ids))

            # expected output put c_ijk: (batch_size, group_size, dim)
            # log_p_ijk: (batch_size, group_size)

            # scores_ijk_sum = scores_ijk + scores_ij_sum  # (batch_size, combination_size)
            scores_ijk_sum = scores_ijk # NOTE: if single step composition is used

            a_ij = F.softmax(scores_ijk_sum, dim=-1)
            # (batch_size, combination_size) => (num_spans, num_split_point)

            # apply gumbel softmax
            c_ij = torch.einsum("ij,ijk->ik", a_ij, c_ijk)
            h_ij_next, _ = h_ij.max(dim=-1)  # (batch_size, group_size)
            h_ij_next = h_ij_next + 1
            h_ij = torch.einsum("ij, ij->i", a_ij, h_ij_next).unsqueeze(1)  # NOTE (batch_size, 1)
            
            scores_ij_sum = torch.einsum("ij, ij->i", a_ij, scores_ijk_sum).unsqueeze(1)

            inside_cache.scatter(target_cache_ids, [self.e_ij_id, self.score_sum_id, self.height_ij],
                                [c_ij, scores_ij_sum, h_ij])
            ## NOTE: non_blocking=False when train on a single ppu (eval)
            if non_blocking:
                splits_orders.append(scores_ijk_sum.argsort(dim=1, descending=True).to('cpu', non_blocking=True))
                split_probabilities.append(a_ij.to('cpu', non_blocking=True))
            else:
                splits_orders.append(scores_ijk_sum.argsort(dim=1, descending=True))
                split_probabilities.append(a_ij)

        if not non_blocking:
            for i in range(len(splits_orders)):
                splits_orders[i] = splits_orders[i].cpu()
            for i in range(len(split_probabilities)):
                split_probabilities[i] = split_probabilities[i].cpu()
        return splits_orders, split_probabilities
        
    
    def outside(self, batch_size, root_ids, inside_cache, outside_groups):
        # initialize tensor cache for outside algorithm
        out_cache_size = inside_cache.capacity - inside_cache.placeholder_num
        outside_cache = TensorCache(0, None, [CacheType.NORMAL, CacheType.NORMAL, CacheType.NORMAL],
                                    [self.input_dim, 1, 1], inside_cache.placeholder_num,
                                    total_cache_size=out_cache_size, 
                                    device=inside_cache.device)
        topdown_e_ij_slot = 0
        topdown_score_slot = 1  # weighted sum for outside scores
        topdown_score_ln_sum = 2  # store log (e^w1 + e^w2 + e^w3), w1, w2, w3 is the calculated outside scores
        
        # (batch_size, dim), add root role embedding
        
        zero_padding = torch.zeros(batch_size, 1, dtype=torch.float, device=self.device)
        neg_padding = torch.zeros((outside_cache.capacity, 1), dtype=torch.float, device=self.device).fill_(-1e20)
        
        # As there is no calcuated outside scores, initialize caches with a huge neg value
        outside_cache.fill(0, outside_cache.capacity, [topdown_score_ln_sum], [neg_padding])
        
        outside_cache.scatter(root_ids.long(), [topdown_e_ij_slot, topdown_score_slot, topdown_score_ln_sum], 
                            [self.outside_root_embedding.unsqueeze(0).repeat(root_ids.shape[0], 1), 
                            zero_padding, zero_padding])

        # run outside according to inside groups
        for target_cache_ids, cache_ids, _ in outside_groups:
            parent_ids = target_cache_ids
            child_ids = cache_ids
            
            parent_ij, parent_ij_score = outside_cache.gather(parent_ids, [topdown_e_ij_slot, topdown_score_slot])

            child_ids_shape = child_ids.shape  # (batch_size, comb_size, 2)
            child_ikj, child_scores = inside_cache.gather(child_ids.flatten(), [self.e_ij_id, self.score_sum_id])
            child_ikj = child_ikj.view(*child_ids.shape, -1)
            child_scores = child_scores.view(*child_ids.shape)  # (batch_size, comb_size, 2)

            out_scores, out_ikj = self.outside_enc(parent_ij, child_ikj, parent_ij_score, child_scores)
            
            dim = out_ikj.shape[-1]

            # weighted sum left and right separately
            weighted_e_ij, weighted_scores, log_ksum_score = \
                outside_cache.gather(child_ids[:, :, 0].flatten(), 
                                     [topdown_e_ij_slot, topdown_score_slot, topdown_score_ln_sum])
            weighted_e_ij = weighted_e_ij.view(*child_ids_shape[:-1], dim)  # (batch_size, comb_size, dim)
            log_ksum_score = log_ksum_score.view(*child_ids_shape[:-1])  # (batch_size, comb_size)
            weighted_scores = weighted_scores.view(*child_ids_shape[:-1])

            # log_p_ijk_mean: (batch_size, comb_size)
            left_k_sum_scores = torch.stack([log_ksum_score, out_scores[:, :, 0]], dim=2)  # (batch_size, comb_size, 2)
            left_k_weights = F.softmax(left_k_sum_scores, dim=2)
            left_weighted_e_ij = left_k_weights[:, :, 0].unsqueeze(2) * weighted_e_ij + \
                                    left_k_weights[:, :, 1].unsqueeze(2) * out_ikj[:, :, 0, :]
            left_weighted_scores = left_k_weights[:, :, 0] * weighted_scores + \
                                    left_k_weights[:, :, 1] * out_scores[:, :, 0]

            # (batch_size, comb_size, dim)
            left_k_sum_scores = left_k_sum_scores.logsumexp(dim=2, keepdim=True)

            left_weighted_e_ij = left_weighted_e_ij.view(-1, dim)
            left_weighted_scores = left_weighted_scores.view(-1, 1)
            left_k_sum_scores = left_k_sum_scores.view(-1, 1)

            outside_cache.scatter(child_ids[:, :, 0].flatten().long(), 
                                  [topdown_e_ij_slot, topdown_score_slot, topdown_score_ln_sum], 
                                  [left_weighted_e_ij, left_weighted_scores, left_k_sum_scores])
            
            weighted_e_ij, weighted_scores, log_ksum_score = \
                outside_cache.gather(child_ids[:, :, 1].flatten(), 
                                     [topdown_e_ij_slot, topdown_score_slot, topdown_score_ln_sum])
            weighted_e_ij = weighted_e_ij.view(*child_ids_shape[:-1], dim)  # (batch_size, comb_size, dim)
            log_ksum_score = log_ksum_score.view(*child_ids_shape[:-1])  # (batch_size, comb_size)
            weighted_scores = weighted_scores.view(*child_ids_shape[:-1])

            right_k_sum_scores = torch.stack([log_ksum_score, out_scores[:, :, 1]], dim=2)  # (batch_size, comb_size, 2)
            right_k_weights = F.softmax(right_k_sum_scores, dim=2)
            right_weighted_e_ij = right_k_weights[:, :, 0].unsqueeze(2) * weighted_e_ij + \
                                    right_k_weights[:, :, 1].unsqueeze(2) * out_ikj[:, :, 1, :]
            right_weighted_scores = right_k_weights[:, :, 0] * weighted_scores + \
                                    right_k_weights[:, :, 1] * out_scores[:, :, 1]

            # (batch_size, comb_size, dim)
            right_k_sum_scores = right_k_sum_scores.logsumexp(dim=2, keepdim=True)

            right_weighted_e_ij = right_weighted_e_ij.view(-1, dim)
            right_weighted_scores = right_weighted_scores.view(-1, 1)
            right_k_sum_scores = right_k_sum_scores.view(-1, 1)
            
            outside_cache.scatter(child_ids[:, :, 1].flatten().long(), 
                                  [topdown_e_ij_slot, topdown_score_slot, topdown_score_ln_sum], 
                                  [right_weighted_e_ij, right_weighted_scores, right_k_sum_scores])

        return outside_cache

    def parser_loss(self, scores, chunk_masks, span_masks, splits):
        # split_masks: (batch_size, L - 1, L - 1)
        # split points: (batch_size, L - 1)
        split_masks = span_masks
        split_points = splits
        L = scores.shape[1]
        attention_mask = chunk_masks


        assert len(attention_mask.shape) == 2
        scores.masked_fill_(attention_mask[:, 1: L + 1] == 0, float('-inf'))
        scores = scores.unsqueeze(1).repeat(1, L, 1)
        scores.masked_fill_(split_masks[:, :L, :L] == 0, float('-inf'))  # (batch_size, L - 1, L - 1)

        return F.cross_entropy(scores.transpose(1, 2).float(), split_points[:, :L], ignore_index=-1)  

    def forward(self, 
                parser_inputs,
                input_ids,
                parser_noise=1.0,
                masks=None,
                group_ids=None,
                padded_group_ids=None,
                gpt_masks=None,
                gpt_gather_ids=None,
                atom_spans:List[List[Tuple[int]]] = None,
                span_ids=None, # deprecated to zeros
                gpt_tgt=None,
                inbatch_sampling=None,
                **kwargs):
        # get merge_trajectory from parser
        merge_trajectory, split_scores  = self.parser(**parser_inputs, noise_coeff=parser_noise)
        seq_lens = torch.sum(masks, dim=1,
                             dtype=torch.int)  # (batch_size)
        ## move inputs to cpu
        seq_lens_np = seq_lens.to('cpu', non_blocking=True)
        merge_trajectory = merge_trajectory.to('cpu', non_blocking=True)
        
        batch_size = input_ids.shape[0]
        flatten_input_ids, input_embedding = self.initialize_embeddings(
            input_ids, seq_lens)
        ids_num = flatten_input_ids.shape[0]
        input_cache_ids = torch.arange(SPECIAL_TOKEN_NUM, 
                                       SPECIAL_TOKEN_NUM + ids_num).to(self.device)
        
        inside_cache = self.create_tensor_cache(seq_lens_np)
        inside_cache.scatter(input_cache_ids, [self.e_ij_id], [input_embedding])

        tables = CPPChartTableManager(seq_lens_np.data.numpy(), self.window_size, merge_trajectory.data.numpy(),
                                      inside_cache.placeholder_num, inside_cache.detach_offset,
                                      group_ids=group_ids, span_ids=span_ids)
        target_cache_ids, span_ids, cache_ids, detach_cache_ids = \
                tables.construct_inside_groups(self.device)
    
        root_ids = tables.root_ids

        splits_orders, split_probabilities = self.inside(inside_cache,
                                    zip(target_cache_ids, span_ids, cache_ids, detach_cache_ids))
        root_embeddings = inside_cache.gather(root_ids, [self.e_ij_id])[0] # inside root 
        padding = self.embedding(torch.tensor([self.pad_token_id], device=self.device))
        root_embeddings = torch.cat((root_embeddings, padding), dim=0)
        
        max_chunk_token_num = parser_inputs['input_ids'].shape[1]
        span_masks, splits = tables.best_trees(splits_orders, group_ids, max_chunk_token_num, atom_spans=atom_spans) 
        span_masks = span_masks.to(self.device, non_blocking=True)
        splits = splits.to(self.device, non_blocking=True)

        outside_cache = self.outside(batch_size, root_ids, inside_cache,
                                    zip(reversed(target_cache_ids), reversed(cache_ids), reversed(detach_cache_ids))
                                    )

        results = {}
        # parser loss
        parser_loss = self.parser_loss(split_scores, parser_inputs["attention_mask"],
                                       span_masks, splits)
        results['parser_loss'] = parser_loss

        outside_token_embeddings = outside_cache.gather(input_cache_ids, [self.e_ij_id])[0]
        lm_logits = self.classifier(outside_token_embeddings)
        lm_loss = F.cross_entropy(lm_logits, flatten_input_ids)
        results['lm_loss'] = lm_loss

        return results

    def inside_inference(self, 
                        parser_inputs,
                        input_ids,
                        masks=None,
                        group_ids=None,
                        atom_spans:List[List[Tuple[int]]] = None,
                        span_ids=None,
                        **kwargs):
        # get merge_trajectory from parser
        merge_trajectory, split_scores = self.parser(**parser_inputs, noise_coeff=0.0)
        seq_lens = torch.sum(masks, dim=1,
                             dtype=torch.int)  # (batch_size)

        ## move inputs to cpu
        seq_lens_np = seq_lens.to('cpu')
        merge_trajectory = merge_trajectory.to('cpu')
        
        batch_size = input_ids.shape[0]
        flatten_input_ids, input_embedding = self.initialize_embeddings(
            input_ids, seq_lens)
        ids_num = flatten_input_ids.shape[0]
        input_cache_ids = torch.arange(SPECIAL_TOKEN_NUM, 
                                       SPECIAL_TOKEN_NUM + ids_num).to(self.device)
        
        inside_cache = self.create_tensor_cache(seq_lens_np)
        inside_cache.scatter(input_cache_ids, [self.e_ij_id], [input_embedding])
        ## NOTE: span_ids=0 when NO_VOCAB_HIT is handled inside cpp code
        tables = CPPChartTableManager(seq_lens_np.data.numpy(), self.window_size, merge_trajectory.data.numpy(),
                                      inside_cache.placeholder_num, inside_cache.detach_offset,
                                      group_ids=group_ids, span_ids=span_ids)
        target_cache_ids, span_ids, cache_ids, detach_cache_ids = \
                tables.construct_inside_groups(self.device)
    
        splits_orders, split_probabilities = self.inside(inside_cache,
                                                         zip(target_cache_ids, span_ids, cache_ids, detach_cache_ids),
                                                         non_blocking=False)
        # after inside: prepare best trees for span loss
        max_chunk_token_num = parser_inputs['input_ids'].shape[1]
        span_masks, splits = tables.best_trees(splits_orders, group_ids, max_chunk_token_num, atom_spans=atom_spans) 
        splits = splits.detach().clone() 
        return splits