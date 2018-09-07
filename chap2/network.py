from util import PAD

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(model, input, optimizer=None, is_train=True):
    """lossを計算するための関数

    is_train=Trueならモデルをtrainモードに、
    is_train=Falseならモデルをevaluationモードに設定します

    :param model: 学習させるモデル
    :param input: モデルへの入力
    :param optimizer: optimizer
    :param is_train: bool, モデルtrainさせるか否か
    """
    model.train(is_train)

    # lossを計算します。
    loss = model(*input)

    if is_train:
        # .backward()を実行する前にmodelのparameterのgradientを全て0にセットします
        optimizer.zero_grad()
        # parameterのgradientを計算します。
        loss.backward()
        # parameterのgradientを用いてparameterを更新します。
        optimizer.step()

    return loss.item()


class SGNS(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        """
        :param vocab_size: int, 語彙の総数
        :param embedding_size: int, 単語埋め込みベクトルの次元
        """
        super(SGNS, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # 入力単語の埋め込み層
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        # 出力単語の埋め込み層
        self.o_embedding = nn.Embedding(self.vocab_size, self.embedding_size)

    def forward(self, batch_X, batch_Y, batch_N):
        """
        :param batch_x: torch.Tensor(dtype=torch.long), (batch_size,)             # NOQA
        :param batch_y: torch.Tensor(dtype=torch.long), (batch_size, window*2)    # NOQA
        :param batch_n: torch.Tensor(dtype=torch.long), (batch_size, n_negative)  # NOQA
        """
        # (batch_size, embedding_size, 1)
        embed_X = self.embedding(batch_X).unsqueeze(2)
        # (batch_size, window*2, embedding_size)
        embed_Y = self.o_embedding(batch_Y)
        # (batch_size, n_negative, embedding_size)
        embed_N = self.o_embedding(batch_N).neg()
        # (batch_size, window*2)
        loss_Y = torch.bmm(embed_Y, embed_X).squeeze().sigmoid().log()
        loss_Y = loss_Y * (batch_Y != PAD).float()  # (batch_size, window*2)
        loss_Y = loss_Y.sum(1)  # (batch_size,)
        loss_N = torch.bmm(embed_N, embed_X).squeeze(
        ).sigmoid().log().sum(1)  # (batch_size,)
        return -(loss_Y + loss_N).mean()
