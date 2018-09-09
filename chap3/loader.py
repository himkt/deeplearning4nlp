from sklearn.utils import shuffle
import torch


random_state = 42


def pad_seq(seq, max_length, pad_value):
    # 系列(seq)が指定の文長(max_length)になるように末尾をパディングする
    res = seq + [pad_value for i in range(max_length - len(seq))]
    return res


class DataLoader(object):

    def __init__(self, X, Y, batch_size, device='cpu', shuffle=False):
        """
        :param X: list, 入力言語の文章（単語IDのリスト）のリスト
        :param Y: list, 出力言語の文章（単語IDのリスト）のリスト
        :param batch_size: int, バッチサイズ
        :param shuffle: bool, サンプルの順番をシャッフルするか否か
        """
        self.data = list(zip(X, Y))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_index = 0
        self.device = device

        self.reset()

    def reset(self):
        if self.shuffle:  # サンプルの順番をシャッフルする
            self.data = shuffle(self.data, random_state=random_state)
        self.start_index = 0  # ポインタの位置を初期化する

    def __iter__(self):
        return self

    def __next__(self):
        # ポインタが最後まで到達したら初期化する
        if self.start_index >= len(self.data):
            self.reset()
            raise StopIteration()

        # バッチを取得
        seqs_X, seqs_Y = zip(
            *self.data[self.start_index:self.start_index+self.batch_size])
        # 入力系列seqs_Xの文章の長さ順（降順）に系列ペアをソートする
        seq_pairs = sorted(zip(seqs_X, seqs_Y),
                           key=lambda p: len(p[0]), reverse=True)
        seqs_X, seqs_Y = zip(*seq_pairs)
        # 短い系列の末尾をパディングする
        # 後述のEncoderのpack_padded_sequenceでも用いる
        lengths_X = [len(s) for s in seqs_X]
        lengths_Y = [len(s) for s in seqs_Y]
        max_length_X = max(lengths_X)
        max_length_Y = max(lengths_Y)
        padded_X = [pad_seq(s, max_length_X, 1) for s in seqs_X]
        padded_Y = [pad_seq(s, max_length_Y, 1) for s in seqs_Y]
        # tensorに変換し、転置する
        batch_X = torch.tensor(padded_X, dtype=torch.long,
                               device=self.device).transpose(0, 1)
        batch_Y = torch.tensor(padded_Y, dtype=torch.long,
                               device=self.device).transpose(0, 1)

        # ポインタを更新する
        self.start_index += self.batch_size

        return batch_X, batch_Y, lengths_X
