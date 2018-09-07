from vocab import UNK
from vocab import PAD


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
