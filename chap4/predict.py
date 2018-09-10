import argparse
import pathlib
import numpy
from util import load_data
from util import sentence_to_ids
from loader import DataLoader

import sys
import csv
import yaml
import torch
import common

from transformer import Transformer


def test(model, src, max_length=20):
    # 学習済みモデルで系列を生成する
    model.eval()

    src_seq, src_pos = src
    batch_size = src_seq.size(0)
    enc_output, enc_slf_attns = model.encoder(src_seq, src_pos)

    tgt_seq = torch.LongTensor(batch_size, 1).fill_(common.BOS).to(common.device)  # NOQA
    tgt_pos = torch.arange(1).unsqueeze(0).repeat(batch_size, 1)
    tgt_pos = tgt_pos.type(torch.LongTensor).to(common.device)

    # 時刻ごとに処理
    for t in range(1, max_length+1):
        dec_output, dec_slf_attns, dec_enc_attns = model.decoder(
            tgt_seq, tgt_pos, src_seq, enc_output)
        dec_output = model.tgt_word_proj(dec_output)
        out = dec_output[:, -1, :].max(dim=-1)[1].unsqueeze(1)
        # 自身の出力を次の時刻の入力にする
        tgt_seq = torch.cat([tgt_seq, out], dim=-1)
        tgt_pos = torch.arange(t+1).unsqueeze(0).repeat(batch_size, 1)
        tgt_pos = tgt_pos.type(torch.LongTensor).to(common.device)

    return tgt_seq[:, 1:], enc_slf_attns, dec_slf_attns, dec_enc_attns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()
    model_dir = pathlib.Path(args.model_dir)

    args_path = model_dir / 'args.yml'
    params = yaml.load(open(args_path.as_posix()))

    data_path = model_dir / 'data.npz'
    data_npz = numpy.load(data_path.as_posix())

    vocab_X = data_npz['vocab_X'].item()
    vocab_Y = data_npz['vocab_Y'].item()

    model_path = model_dir / f'model_{args.epoch:03d}.pth'
    model = Transformer(**params)
    model.to(common.device)
    model.load_state_dict(torch.load(model_path.as_posix()))
    print(f'loaded model from {model_path}', file=sys.stderr)

    test_X = []
    test_max_length = 0
    for sentence in load_data('../data/chap4/test.en'):
        test_X.append(sentence_to_ids(vocab_X, sentence))

    test_dataloader = DataLoader(test_X, test_X, params['batch_size'], shuffle=False)  # NOQA

    pred_Y = []
    for batch in test_dataloader:
        batch_X, _ = batch
        preds, *_ = test(model, batch_X)
        preds = preds.data.cpu().numpy().tolist()
        preds = [pred[:pred.index(common.EOS)] if common.EOS in pred else pred
                 for pred in preds]
        pred_y = [[vocab_Y.id2word[_id] for _id in pred] for pred in preds]  # NOQA
        pred_Y += pred_y

    with open('submission.csv', 'w') as f:
        writer = csv.writer(f, delimiter=' ', lineterminator='\n')
        writer.writerows(pred_Y)
