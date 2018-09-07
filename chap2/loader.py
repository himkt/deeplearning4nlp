from util import pad_seq
import torch
import numpy


class DataLoaderSGNS(object):
    def __init__(self, text, batch_size, device, negative_table,
                 window=5, n_negative=5):
        """
        :param text: list of list of int, 単語をIDに変換したデータセット
        :param batch_size: int, ミニバッチのサイズ
        :param window: int, 周辺単語と入力単語の最大距離
        :param n_negative: int, 負例の数
        :param weights: numpy.ndarray, Negative Samplingで使う確率分布
        """
        self.text = text
        self.batch_size = batch_size
        self.window = window
        self.n_negative = n_negative
        self.negative_table = negative_table
        self.s_pointer = 0  # 文のポインタ
        self.w_pointer = 0  # 単語のポインタ
        self.max_s_pointer = len(text)
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        batch_X = []
        batch_Y = []
        batch_N = []  # 負例

        table_size, = self.negative_table.shape
        while len(batch_X) < self.batch_size:
            if self.w_pointer % 10000 == 0:
                print(self.s_pointer, self.w_pointer, len(self.text[self.s_pointer]))
            sen = self.text[self.s_pointer]
            start = max(0, self.w_pointer - self.window)
            word_X = sen[self.w_pointer]
            word_Y = sen[start:self.w_pointer] + \
                sen[self.w_pointer + 1:self.w_pointer + self.window + 1]
            word_Y = pad_seq(word_Y, self.window * 2)
            batch_X.append(word_X)
            batch_Y.append(word_Y)

            # 多項分布で負例をサンプリング
            # 実装を簡略化するために、正例の除去は行っていません
            # batch_N.append(negative_samples.unsqueeze(0))  # (1, n_negative)
            n_idxs = numpy.random.randint(low=0, high=table_size, size=self.n_negative)
            negative_sample = self.negative_table[n_idxs].reshape(1, -1)
            negative_sample = torch.tensor(negative_sample, dtype=torch.long)
            batch_N.append(negative_sample)

            self.w_pointer += 1
            if self.w_pointer >= len(sen):
                self.w_pointer = 0
                self.s_pointer += 1
                if self.s_pointer >= self.max_s_pointer:
                    self.s_pointer = 0
                    raise StopIteration

        batch_X = torch.tensor(batch_X, dtype=torch.long, device=self.device)
        batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=self.device)
        # (batch_size, n_negative)
        batch_N = torch.cat(batch_N, dim=0).to(self.device)

        return batch_X, batch_Y, batch_N
