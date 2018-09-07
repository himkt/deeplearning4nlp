from vocab import Vocab

import argparse
import torch
import numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--vocab')
    args = parser.parse_args()

    word_pairs = []
    with open("../data/chap2/sample_submission.csv", "r") as fin:
        for line in fin:
            line = line.strip().split(",")
            word1 = line[0]
            word2 = line[1]
            word_pairs.append([word1, word2])

    model = torch.load(args.model)
    z = numpy.sqrt((model * model).sum(axis=1))
    model /= z.reshape(-1, 1)
    vocab = Vocab()
    vocab.load(args.vocab)

    for word1, word2 in word_pairs:
        wordid1 = vocab.word2id[word1]
        wordid2 = vocab.word2id[word2]
        score = model[wordid1].dot(model[wordid2])
        print(f'{word1},{word2},{str(score)}')
