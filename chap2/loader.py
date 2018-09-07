from util import pad_seq
import torch


class DataLoaderCBOW(object):
    """CBOWのデータローダー"""

    def __init__(self, text, batch_size, device, window=3):
        """
        :param text: list of list of int, 単語をIDに変換したデータセット
        :param batch_size: int, ミニバッチのサイズ
        :param window: int, 周辺単語とターゲットの単語の最大距離
        """
        self.text = text
        self.batch_size = batch_size
        self.window = window
        self.s_pointer = 0  # データセット上を走査する文単位のポインタ
        self.w_pointer = 0  # データセット上を走査する単語単位のポインタ
        self.max_s_pointer = len(text)  # データセットに含まれる文の総数
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        batch_X = []
        batch_Y = []
        while len(batch_X) < self.batch_size:
            # 走査する対象の文
            sen = self.text[self.s_pointer]

            # 予測すべき単語
            word_Y = sen[self.w_pointer]

            # 入力となる単語群を取得
            start = max(0, self.w_pointer - self.window)
            word_X = sen[start:self.w_pointer] + \
                sen[self.w_pointer + 1:self.w_pointer + self.window + 1]
            word_X = pad_seq(word_X, self.window * 2)

            batch_X.append(word_X)
            batch_Y.append(word_Y)
            self.w_pointer += 1

            if self.w_pointer >= len(sen):
                # 文を走査し終わったら次の文の先頭にポインタを移行する
                self.w_pointer = 0
                self.s_pointer += 1
                if self.s_pointer >= self.max_s_pointer:
                    # 全ての文を走査し終わったら終了する
                    self.s_pointer = 0
                    raise StopIteration

        # データはtorch.Tensorにする必要があります。dtype, deviceも指定します。
        batch_X = torch.tensor(batch_X, dtype=torch.long, device=self.device)
        batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=self.device)

        return batch_X, batch_Y


class DataLoaderSG(object):
    """Skipgramのためのデータローダー"""

    def __init__(self, text, batch_size, device, window=3):
        """
        :param text: list of list of int, 単語をIDに変換したデータセット
        :param batch_size: int, ミニバッチのサイズ
        :param window: int, 周辺単語と入力単語の最大距離
        """
        self.text = text
        self.batch_size = batch_size
        self.window = window
        self.s_pointer = 0  # データセット上を走査する文単位のポインタ
        self.w_pointer = 0  # データセット上を走査する単語単位のポインタ
        self.max_s_pointer = len(text)  # データセットに含まれる文の総数
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        batch_X = []
        batch_Y = []

        while len(batch_X) < self.batch_size:
            sen = self.text[self.s_pointer]

            # Skipgramでは入力が1単語
            word_X = sen[self.w_pointer]

            # 出力は周辺単語
            start = max(0, self.w_pointer - self.window)
            word_Y = sen[start:self.w_pointer] + \
                sen[self.w_pointer + 1:self.w_pointer + self.window + 1]
            word_Y = pad_seq(word_Y, self.window * 2)

            '''
            # Skipgramでは入力が1単語
            word_X = # WRITE ME

            # 出力は周辺単語
            start = # WRITE ME
            word_Y = # WRITE ME
            word_Y = # WRITE ME, paddingが必要
            '''

            batch_X.append(word_X)
            batch_Y.append(word_Y)
            self.w_pointer += 1

            if self.w_pointer >= len(sen):
                self.w_pointer = 0
                self.s_pointer += 1
                if self.s_pointer >= self.max_s_pointer:
                    self.s_pointer = 0
                    raise StopIteration

        batch_X = torch.tensor(batch_X, dtype=torch.long, device=self.device)
        batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=self.device)

        return batch_X, batch_Y


class DataLoaderSGNS(object):
    def __init__(self, text, batch_size, device,
                 window=3, n_negative=5, weights=None):
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
        self.weights = None
        if weights is not None:
            # negative samplingに使う確率分布
            self.weights = torch.FloatTensor(weights)
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
        while len(batch_X) < self.batch_size:
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
            negative_samples = torch.multinomial(self.weights, self.n_negative)
            batch_N.append(negative_samples.unsqueeze(0))  # (1, n_negative)

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
