from model.topdown_parser import BasicParser
import torch.nn as nn
import torch
import numpy as np
from .Llama_flash_attn import Transformer, ModelArgs


class TransformerParser(BasicParser):
    def __init__(self, config) -> None:
        super().__init__()
        self.legacy_mode = False
        self.hidden_dim = config.parser_hidden_dim
        self.input_dim = config.parser_input_dim

        self.score_mlp = nn.Sequential(nn.Linear(2 * self.input_dim, self.hidden_dim),
                                       nn.GELU(),
                                       nn.Dropout(config.hidden_dropout_prob),
                                       nn.Linear(self.hidden_dim, 1))
        args = ModelArgs(config.parser_input_dim, config.parser_num_layers, config.parser_nhead,
                         config.vocab_size, max_seq_len=config.parser_max_len, apply_norm=True)
        self.encoder = Transformer(args)

    def _generate_flatten_input_ids(self, input_ids, attn_mask, group_ids):
        seq_lens = attn_mask.sum(dim=1).cpu().data.numpy()
        batch_size = group_ids[-1] + 1
        group_lengths = [0] * batch_size
        for sent_id, group_id in enumerate(group_ids):
            group_lengths[group_id] += seq_lens[sent_id]

        max_length = max(group_lengths)

        prev_group_id = -1
        flatten_ids = input_ids.new_zeros((batch_size, max_length))
        flatten_masks = attn_mask.new_zeros([batch_size, max_length])
        for sent_id, group_id in enumerate(group_ids):
            if prev_group_id != group_id:
                offset = 0
                prev_group_id = group_id
            flatten_ids[group_id, offset: offset + seq_lens[sent_id]] = input_ids[sent_id, :seq_lens[sent_id]]
            flatten_masks[group_id, offset: offset + seq_lens[sent_id]] = 1
            offset += seq_lens[sent_id]
        return flatten_ids, flatten_masks, seq_lens

    def _recover_score_chunks(self, org_shape, scores, seq_lens, group_ids):
        rev_scores = scores.new_zeros((org_shape[0], org_shape[1] - 1))  # (N, L)
        offset = 0
        prev_group_id = -1
        for sent_id, group_id in enumerate(group_ids):
            if group_id != prev_group_id:
                prev_group_id = group_id
                offset = 0
            sent_len = seq_lens[sent_id]
            rev_scores[sent_id, : sent_len - 1] = scores[group_id, offset: offset + sent_len - 1]
            offset += sent_len
        return rev_scores

    def _split_point_scores(self, input_ids, attn_mask, group_ids=None):
        # attn_mask: (N, L) recording segment ids
        if attn_mask is None:
            attn_mask = torch.ones_like(input_ids)
        attn_mask = attn_mask.unsqueeze(2) == attn_mask.unsqueeze(1)  # (N, L, L) or (L, L)
        mask = torch.zeros_like(attn_mask, dtype=torch.float)
        mask.masked_fill_(attn_mask == 0, -np.inf)
        N = input_ids.shape[0]
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        outputs = self.encoder(input_ids, attn_mask=mask, position_ids=pos_ids)
        split_logits = torch.cat([outputs[:, :-1, :], outputs[:, 1:, :]], dim=-1)  # (N, L - 1, 2 * dim)
        scores = self.score_mlp(split_logits)
        scores = scores.squeeze(-1)
        return scores