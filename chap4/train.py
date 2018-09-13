import time
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

from nltk import bleu_score

import torch
import torch.nn as nn
import torch.optim as optim

from util import load_data
from util import sentence_to_ids
from loader import DataLoader
from vocab import Vocab
from transformer import Transformer
import pathlib
import common
import argparse
import yaml
import sys
import numpy
import json


def compute_loss(batch_X, batch_Y, model, criterion,
                 optimizer=None, is_train=True):
    # バッチの損失を計算
    model.train(is_train)

    pred_Y = model(batch_X, batch_Y)
    gold = batch_Y[0][:, 1:].contiguous()
    loss = criterion(pred_Y.view(-1, pred_Y.size(2)), gold.view(-1))

    if is_train:  # 訓練時はパラメータを更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    gold = gold.data.cpu().numpy().tolist()
    pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().tolist()

    return loss.item(), gold, pred


def calc_bleu(refs, hyps):
    """
    BLEUスコアを計算する関数
    :param refs: list, 参照訳。単語のリストのリスト
    (例： [['I', 'have', 'a', 'pen'], ...])
    :param hyps: list, モデルの生成した訳。単語のリストのリスト
    (例： [['I', 'have', 'a', 'pen'], ...])
    :return: float, BLEUスコア(0~100)
    """
    refs = [[ref[:ref.index(common.EOS)]] for ref in refs]
    hyps = [hyp[:hyp.index(common.EOS)]
            if common.EOS in hyp else hyp for hyp in hyps]
    return 100 * bleu_score.corpus_bleu(refs, hyps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    params = yaml.load(open(args.config))
    output_dir = pathlib.Path(params['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    word2id = common.word2id

    train_X = load_data('../data/chap4/train.en')
    train_Y = load_data('../data/chap4/train.ja')

    # 訓練データと検証データに分割
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y,
                                                          test_size=0.2,
                                                          random_state=common.random_state)  # NOQA

    vocab_X = Vocab(word2id=word2id)
    vocab_Y = Vocab(word2id=word2id)
    vocab_X.build_vocab(train_X, min_count=common.MIN_COUNT)
    vocab_Y.build_vocab(train_Y, min_count=common.MIN_COUNT)

    vocab_size_X = len(vocab_X.id2word)
    vocab_size_Y = len(vocab_Y.id2word)

    word_dim = params['d_word_vec']
    scale = numpy.sqrt(3.0 / word_dim)

    # src: 英語, tgt: 日本語
    src_syn0 = numpy.random.uniform(-scale, scale, [vocab_size_X, word_dim])
    tgt_syn0 = numpy.random.uniform(-scale, scale, [vocab_size_Y, word_dim])

    # TODO load pre-trained embeddings
    if 'src_word_vec' in params:
        match_word_num = 0
        emb_model = KeyedVectors.load('../data/common/en/glove_200d')
        for word, idx in vocab_X.word2id.items():
            # do not have to lower (all words are lowercases)
            if word in emb_model:
                src_syn0[idx, :] = emb_model.word_vec(word)
                match_word_num += 1

        msg = 'use pre-trained word embeddings'
        msg += f' ({match_word_num} words in vocab)'
        print(msg, file=sys.stderr)

    train_X = [sentence_to_ids(vocab_X, sentence) for sentence in train_X]
    train_Y = [sentence_to_ids(vocab_Y, sentence) for sentence in train_Y]
    valid_X = [sentence_to_ids(vocab_X, sentence) for sentence in valid_X]
    valid_Y = [sentence_to_ids(vocab_Y, sentence) for sentence in valid_Y]

    max_length = 20
    ckpt_path = 'transformer.pth'
    max_length = max_length + 2
    params['n_src_vocab'] = vocab_size_X
    params['n_tgt_vocab'] = vocab_size_Y
    params['max_length'] = max_length

    # DataLoaderやモデルを定義
    train_dataloader = DataLoader(train_X, train_Y, params['batch_size'])
    valid_dataloader = DataLoader(valid_X, valid_Y, params['batch_size'],
                                  shuffle=False)

    args_path = output_dir / 'args.yml'
    yaml.dump(params, open(args_path.as_posix(), 'w'))

    data_fpath = output_dir / 'data.npz'
    numpy.savez(data_fpath.as_posix(),
                vocab_X=vocab_X, vocab_Y=vocab_Y)
    print(f'save data into #{data_fpath}')

    model = Transformer(**params).to(common.device)
    optimizer = optim.Adam(model.get_trainable_parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss(
        ignore_index=common.PAD, size_average=False).to(common.device)

    # 訓練
    best_valid_bleu = 0.

    log_path = output_dir / 'logs.jsonl'
    log_file = open(log_path.as_posix(), 'w')

    for epoch in range(1, params['num_epochs']+1):
        start = time.time()
        train_loss = 0.
        train_refs = []
        train_hyps = []
        valid_loss = 0.
        valid_refs = []
        valid_hyps = []
        # train
        for batch in train_dataloader:
            batch_X, batch_Y = batch
            loss, gold, pred = compute_loss(
                batch_X, batch_Y, model, criterion, optimizer, is_train=True
            )
            train_loss += loss
            train_refs += gold
            train_hyps += pred
        # valid
        for batch in valid_dataloader:
            batch_X, batch_Y = batch
            loss, gold, pred = compute_loss(
                batch_X, batch_Y, model, criterion, is_train=False
            )
            valid_loss += loss
            valid_refs += gold
            valid_hyps += pred
        # 損失をサンプル数で割って正規化
        train_loss /= len(train_dataloader.data)
        valid_loss /= len(valid_dataloader.data)
        # BLEUを計算
        train_bleu = calc_bleu(train_refs, train_hyps)
        valid_bleu = calc_bleu(valid_refs, valid_hyps)

        # validationデータでBLEUが改善した場合にはモデルを保存
        if valid_bleu > best_valid_bleu:
            ckpt = model.state_dict()
            ckpt_path = output_dir / f'model_{epoch:03d}.pth'
            torch.save(ckpt, ckpt_path.as_posix())
            best_valid_bleu = valid_bleu

        elapsed_time = (time.time()-start) / 60
        msg = f'Epoch {epoch:3d} ({elapsed_time:.2f}) train_loss: {train_loss:.2f} train_bleu: {train_bleu:.2f}'  # NOQA
        msg += f' valid_loss: {valid_loss:.2f} valid_bleu: {valid_bleu:.2f}'

        print(msg, file=sys.stderr)
        print('-'*80, file=sys.stderr)

        log_items = {'epoch': epoch, 'train_loss': train_loss, 'train_bleu': train_bleu}  # NOQA
        log_items.update({'valid_loss': valid_loss, 'valid_bleu': valid_bleu})
        print(json.dumps(log_items), file=log_file)

    log_file.close()
