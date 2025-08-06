from collections import Counter, defaultdict
import torch


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return -1e9

class UReca(Metric):
    def __init__(self, eps=1e-8, device=torch.device("cuda")):
        super(UReca, self).__init__()
        self.reca = 0.0
        self.n = 0.0
        self.eps = eps
        self.hit = 0.0 # tp
        self.tgt = 0.0 # tp + fn
        self.device = device


    def __call__(self, pred, gold_list, filter_leaf=True):
        """
            :param pred: list[tuple]
            :param gold_list: list[list[tuple]]
            there could be multiple golds for one pred.
            :return: none
        """
        length = max(gold_list[0],key=lambda x:x[1])[1]
        # removing leaf spans
        if filter_leaf:
            pred = list(filter(lambda x: x[0] != x[1], pred))
        # removing root span
        pred = list(filter(lambda x: not (x[0]==0 and x[1]==length), pred))

        # given multiple golds
        max_reca = (-1, 1)
        for gold in gold_list:
            if filter_leaf:
                gold = list(filter(lambda x: x[0] != x[1], gold))
            gold = list(filter(lambda x: not (x[0]==0 and x[1]==length), gold))
            hit = 0
            tgt = len(gold)
            if tgt > 0:
                for span in gold:
                    if span in pred:
                        hit += 1
                if hit / tgt > max_reca[0] / max_reca[1]:
                    max_reca = (hit, tgt)
        if max_reca[0] != -1:
            self.hit += max_reca[0]
            self.tgt += max_reca[1]
            self.reca += max_reca[0] / max_reca[1]
            self.n += 1
        # return max_reca

    @property
    def sentence_ureca(self):
        return self.reca / self.n

    @property
    def corpus_ureca(self):
        if self.tgt == 0: # no nontrivial gold
            return -1

        corpus_recall = self.hit / self.tgt
        return corpus_recall

    @property
    def score(self):
        return self.sentence_ureca

    def __repr__(self):
        s_reca = f"Sentence Recall: {self.sentence_ureca:6.2%} Corpus Recall: {self.corpus_ureca:6.2%} "
        return s_reca
