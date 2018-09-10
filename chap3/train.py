from sklearn.model_selection import train_test_split
from loader import pad_seq
from loader import DataLoader
from vocab import Vocab
from network import EncoderDecoder
from network import BeamEncoderDecoder
from nltk import bleu_score
from common import word2id
from util import load_data
from util import sentence_to_ids

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


def masked_cross_entropy(logits, target):
    return mce(logits.view(-1, logits.size(-1)), target.view(-1))


def compute_loss(batch_X, batch_Y, lengths_X, model,
                 optimizer=None, is_train=True):
    # 損失を計算する関数
    model.train(is_train)  # train/evalモードの切替え

    # 一定確率でTeacher Forcingを行う
    use_teacher_forcing = is_train and (random.random() < teacher_forcing_rate)
    max_length = batch_Y.size(0)
    # 推論
    # pred_Y = model(batch_X, lengths_X, max_length)
    pred_Y = model(batch_X, lengths_X, max_length,
                   batch_Y, use_teacher_forcing)

    # 損失関数を計算
    loss = masked_cross_entropy(pred_Y.contiguous(), batch_Y.contiguous())

    if is_train:  # 訓練時はパラメータを更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    batch_Y = batch_Y.transpose(0, 1).contiguous().data.cpu().tolist()
    pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().T.tolist()

    return loss.item(), batch_Y, pred


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    params = yaml.load(open(args.config))
    output_dir = pathlib.Path(params['output']['model_fpath'])
    output_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(1)
    random_state = 42

    train_X = load_data('../data/chap3/train.en')
    train_Y = load_data('../data/chap3/train.ja')
    # 演習用にデータサイズを縮小
    train_X = train_X[:len(train_X)//2]
    train_Y = train_Y[:len(train_Y)//2]

    # 訓練データと検証データに分割
    train_X, valid_X, train_Y, valid_Y = train_test_split(
        train_X, train_Y, test_size=0.2, random_state=random_state)

    MIN_COUNT = 2  # 語彙に含める単語の最低出現回数

    # 単語辞書を作成
    vocab_X = Vocab(word2id=word2id)
    vocab_Y = Vocab(word2id=word2id)
    vocab_X.build_vocab(train_X, min_count=MIN_COUNT)
    vocab_Y.build_vocab(train_Y, min_count=MIN_COUNT)

    vocab_size_X = len(vocab_X.id2word)
    vocab_size_Y = len(vocab_Y.id2word)
    print('入力言語の語彙数：', vocab_size_X)
    print('出力言語の語彙数：', vocab_size_Y)

    train_X = [sentence_to_ids(vocab_X, sentence) for sentence in train_X]
    train_Y = [sentence_to_ids(vocab_Y, sentence) for sentence in train_Y]
    valid_X = [sentence_to_ids(vocab_X, sentence) for sentence in valid_X]
    valid_Y = [sentence_to_ids(vocab_Y, sentence) for sentence in valid_Y]

    mce = nn.CrossEntropyLoss(size_average=False,
                              ignore_index=word2id['<PAD>'])

    # ハイパーパラメータの設定
    num_epochs = params['training']['epoch']
    batch_size = params['training']['batch_size']
    lr = params['training']['learning_rate']  # 学習率
    teacher_forcing_rate = params['training']['teacher_forcing_rate']
    # Teacher Forcingを行う確率

    model_args = {
        'input_size': vocab_size_X,
        'output_size': vocab_size_Y,
        'device': device
    }
    model_args.update(params['network'])

    # モデルとOptimizerを定義
    model = EncoderDecoder(**model_args).to(device)
    # model = BeamEncoderDecoder(**model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    args_fpath = output_dir / 'args.yml'
    model_args.pop('device')
    yaml.dump(model_args, open(args_fpath, 'w'))
    print(f'save args into #{args_fpath}')

    data_fpath = output_dir / 'data.npz'
    numpy.savez(data_fpath.as_posix(),
                vocab_X=vocab_X, vocab_Y=vocab_Y)
    print(f'save data into #{data_fpath}')

    # データローダを定義
    train_dataloader = DataLoader(train_X, train_Y, batch_size,
                                  device, shuffle=True)
    valid_dataloader = DataLoader(valid_X, valid_Y, batch_size,
                                  device, shuffle=False)

    # 訓練
    best_valid_bleu = 0.

    for epoch in range(1, num_epochs+1):
        train_loss = 0.
        train_refs = []
        train_hyps = []
        valid_loss = 0.
        valid_refs = []
        valid_hyps = []
        # train
        for batch in train_dataloader:
            batch_X, batch_Y, lengths_X = batch
            loss, gold, pred = compute_loss(
                batch_X, batch_Y, lengths_X, model, optimizer,
                is_train=True)
            train_loss += loss
            train_refs += gold
            train_hyps += pred
        # valid
        for batch in valid_dataloader:
            batch_X, batch_Y, lengths_X = batch
            loss, gold, pred = compute_loss(
                batch_X, batch_Y, lengths_X, model,
                is_train=False
            )
            valid_loss += loss
            valid_refs += gold
            valid_hyps += pred
        # 損失をサンプル数で割って正規化
        train_loss = np.sum(train_loss) / len(train_dataloader.data)
        valid_loss = np.sum(valid_loss) / len(valid_dataloader.data)
        # BLEUを計算
        train_bleu = calc_bleu(train_refs, train_hyps)
        valid_bleu = calc_bleu(valid_refs, valid_hyps)

        msg = f'Epoch #{epoch:04d} train_loss: {train_loss:.2f}'
        msg += f' valid_loss: {valid_loss:.2f} valid_blue: {valid_bleu:.2f}'

        ckpt_fname = f'model_{epoch:03d}.pth'  # 学習済みのモデルを保存するパス
        ckpt_path = output_dir / ckpt_fname
        ckpt_path = ckpt_path.as_posix()

        # validationデータでBLEUが改善した場合にはモデルを保存
        if valid_bleu > best_valid_bleu:
            ckpt = model.state_dict()
            torch.save(ckpt, ckpt_path)
            best_valid_bleu = valid_bleu
            msg += f' (Updated)'

        print(msg)

        print('-'*80)
