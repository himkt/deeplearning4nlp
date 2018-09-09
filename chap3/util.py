from sklearn.model_selection import train_test_split
from loader import pad_seq
from loader import DataLoader
from vocab import Vocab
from network import EncoderDecoder
from nltk import bleu_score
from common import word2id

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
import random
import numpy
import yaml


def load_data(file_path):
    # テキストファイルからデータを読み込む関数
    data = []
    for line in open(file_path, encoding='utf-8'):
        words = line.strip().split()  # スペースで単語を分割
        data.append(words)
    return data


def sentence_to_ids(vocab, sentence):
    # 単語(str)のリストをID(int)のリストに変換する関数
    ids = [vocab.word2id.get(word, vocab.word2id['<UNK>'])
           for word in sentence]
    ids += [vocab.word2id['<EOS>']]  # EOSを加える
    return ids


def calc_bleu(refs, hyps):
    """
    BLEUスコアを計算する関数
    :param refs: list, 参照訳。単語のリストのリスト (例： [['I', 'have', 'a', 'pen'], ...])
    :param hyps: list, モデルの生成した訳。
    単語のリストのリスト (例： [['I', 'have', 'a', 'pen'], ...])
    :return: float, BLEUスコア(0~100)
    """
    refs = [[ref[:ref.index(word2id['<EOS>'])]] for ref in refs]
    hyps = [hyp[:hyp.index(word2id['<EOS>'])]
            if word2id['<EOS>'] in hyp else hyp for hyp in hyps]
    return 100 * bleu_score.corpus_bleu(refs, hyps)
