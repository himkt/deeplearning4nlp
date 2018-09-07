from vocab import UNK
from vocab import PAD

import numpy


def load_data(data_fpath):
    with open(data_fpath) as f:
        line = f.readline()
        line = line.strip().split()
    return line


def sentence_to_ids(vocab, sen):
    """
    単語のリストをIDのリストに変換する関数

    :param vocab: class `Vocab` object
    :param sen: list of str, 文を分かち書きして得られた単語のリスト
    :return out: list of int, 単語IDのリスト
    """
    out = [vocab.word2id.get(word, UNK) for word in sen]
    return out


def pad_seq(seq, max_length):
    """Paddingを行う関数

    :param seq: list of int, 単語のインデックスのリスト
    :param max_length: int, バッチ内の系列の最大長
    :return seq: list of int, 単語のインデックスのリスト
    """
    seq += [PAD for i in range(max_length - len(seq))]
    return seq


def init_negative_table(frequency, negative_alpha, table_length):
    z = numpy.sum(numpy.power(frequency, negative_alpha))
    negative_table = numpy.zeros(table_length, dtype=numpy.int32)
    begin_index = 0
    for word_id, freq in enumerate(frequency):
        c = numpy.power(freq, negative_alpha)
        end_index = begin_index + int(c * table_length / z) + 1
        negative_table[begin_index:end_index] = word_id
        begin_index = end_index
    return negative_table
