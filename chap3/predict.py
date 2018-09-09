import argparse
import pathlib
import random
import numpy
from sklearn.utils import shuffle
from nltk import bleu_score
from util import load_data
from util import sentence_to_ids
from loader import DataLoader
from loader import pad_seq
from common import word2id

import sys
import csv
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from network import EncoderDecoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()
    model_dir = pathlib.Path(args.model_dir)

    args_path = model_dir / 'args.yml'
    args_params = yaml.load(open(args_path.as_posix()))

    data_path = model_dir / 'data.npz'
    data_npz = numpy.load(data_path.as_posix())

    vocab_X = data_npz['vocab_X'].item()
    vocab_Y = data_npz['vocab_Y'].item()

    model_path = model_dir / 'model.pth'
    model = EncoderDecoder(**args_params)
    model.load_state_dict(torch.load(model_path.as_posix()))
    print(f'loaded model from {model_path}', file=sys.stderr)

    test_X = []
    test_max_length = 0
    for sentence in load_data('../data/chap3/test.en'):
        test_X.append(sentence_to_ids(vocab_X, sentence))
        test_max_length = max(test_max_length, len(test_X[-1]))

    test_dataloader = DataLoader(test_X, test_X, 1,
                                 shuffle=False)

    pred_Y = []
    for batch in test_dataloader:
        batch_X, _, lengths_X = batch
        pred = model(batch_X, lengths_X, max_length=test_max_length)
        pred = pred.max(dim=-1)[1].view(-1).data.cpu().numpy().tolist()
        if word2id['<EOS>'] in pred:
            pred = pred[:pred.index(word2id['<EOS>'])]
        pred_y = [vocab_Y.id2word[_id] for _id in pred]
        pred_Y.append(pred_y)

    with open('./submission.csv', 'w') as f:
        writer = csv.writer(f, delimiter=' ', lineterminator='\n')
        writer.writerows(pred_Y)
