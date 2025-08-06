# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

from torch import nn
import torch


class R2D2Base(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        ## NOTE ext_vocab 2.0: basic vocab and ext_vocab should be concatenated
        self.ext_vocab_size = getattr(config, 'ext_vocab_size', 0)
        self.vocab_size = self.ext_vocab_size if self.ext_vocab_size>0 else config.vocab_size
        ##
        self.input_dim = config.hidden_size
        self.hidden_dim = config.intermediate_size
        self.window_size = config.window_size
        ## NOTE: need to reserve the first entry in vocab
        self.embedding = nn.Embedding(self.vocab_size, self.input_dim)
        self.classifier = nn.Linear(self.input_dim, self.vocab_size, bias=False) # NOTE: delete if only ar loss
        ##
        self.tie_decoder = False # getattr(config, 'tie_decoder', False)
        print("tie_decoder is always false")
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

        self._initialize_weights()
        if self.tie_decoder:
            self._tie_weights()

    def _tie_weights(self):
        self.classifier.weight = self.embedding.weight

    def _initialize_weights(self):
        self.embedding.weight.data.normal_(mean=0, std=0.02)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def eos_vec(self):
        return self.embedding(torch.tensor([self.eos_token_id]).to(self.device)).squeeze(0)

    @property
    def bos_vec(self):
        return self.embedding(torch.tensor([self.bos_token_id]).to(self.device)).squeeze(0)

    def from_pretrain(self, model_path, strict=True):
        state_dict = torch.load(model_path, map_location=lambda a, b: a)
        transfered_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            transfered_state_dict[new_k] = v
        self.load_state_dict(transfered_state_dict, strict=strict)