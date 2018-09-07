from loader import DataLoaderSGNS
from network import compute_loss
from network import SGNS
from util import sentence_to_ids
from util import load_data
from util import init_negative_table
from vocab import Vocab

import torch
import torch.optim as optim
import pathlib
import argparse
import numpy
import time
import tqdm
import yaml


MIN_COUNT = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./cbow.config')
    args = parser.parse_args()

    params = yaml.load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = load_data("../data/chap2/text8")
    vocab = Vocab()
    vocab.build_vocab([text], min_count=MIN_COUNT)
    vocab_size = len(vocab.word2id)
    print("語彙数:", len(vocab.word2id))
    id_text = [sentence_to_ids(vocab, sen) for sen in tqdm.tqdm([text])]

    negative_table_size = 1_000_000
    weights = numpy.power([0, 0] + list(vocab.raw_vocab.values()), 0.75)
    weights = weights / weights.sum()
    negative_table = init_negative_table(weights, 0.75, negative_table_size)
    model = SGNS(vocab_size, params['embedding_size']).to(device)
    dataloader = DataLoaderSGNS(id_text, params['batch_size'],
                                device, negative_table, n_negative=5)

    output_dir = pathlib.Path(params['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    vocab.save(output_dir.as_posix())

    print(model)
    optimizer = optim.Adam(model.parameters())
    start_at = time.time()
    accum_loss = 0.0

    for batch_id, batch in enumerate(dataloader):
        loss = compute_loss(model, batch, optimizer=optimizer, is_train=True)
        accum_loss += loss

        if batch_id % 1000 == 0:
            end_at = time.time()
            elapsed = end_at - start_at
            print(f'Batch #{batch_id}, Loss:{accum_loss:.2f}, Elapsed:{elapsed:.2f}')

            model_fname = params['algorithm'] + f'{batch_id:08d}.pth'
            model_fpath = pathlib.PurePath(output_dir, model_fname).as_posix()
            embedding_matrix = model.embedding.weight.data.cpu().numpy()
            torch.save(embedding_matrix, model_fpath)
            print(f'saved model into {model_fpath}')
            start_at = end_at
            accum_loss = 0
