from loader import DataLoaderCBOW
from loader import DataLoaderSGNS
from loader import DataLoaderSG
from network import compute_loss
from network import Skipgram
from network import CBOW
from network import SGNS
from util import sentence_to_ids
from util import load_data
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
    text = load_data("./data/text8")
    vocab = Vocab()
    vocab.build_vocab([text], min_count=MIN_COUNT)
    vocab_size = len(vocab.word2id)
    print("語彙数:", len(vocab.word2id))
    id_text = [sentence_to_ids(vocab, sen) for sen in tqdm.tqdm([text])]

    if params['algorithm'] == 'cbow':
        model = CBOW(vocab_size, params['embedding_size']).to(device)
        dataloader = DataLoaderCBOW(id_text, params['batch_size'], device)

    elif params['algorithm'] == 'sg':
        model = Skipgram(vocab_size, params['embedding_size']).to(device)
        dataloader = DataLoaderSG(id_text, params['batch_size'], device)

    elif params['algorithm'] == 'sgns':
        # negative samplingに使う確率分布
        weights = numpy.power([0, 0] + list(vocab.raw_vocab.values()), 0.75)
        weights = weights / weights.sum()
        model = SGNS(vocab_size, params['embedding_size']).to(device)
        dataloader = DataLoaderSGNS(id_text, params['batch_size'], device,
                                    n_negative=5, weights=weights)

    output_dir = pathlib.Path(params['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    print(model)
    optimizer = optim.Adam(model.parameters())
    start_at = time.time()

    for batch_id, batch in enumerate(dataloader):
        loss = compute_loss(model, batch, optimizer=optimizer, is_train=True)
        if batch_id % 100 == 0:
            print("batch:{}, loss:{:.4f}".format(batch_id, loss))
            model_fname = params['algorithm'] + f'{batch_id:03d}.pth'
            model_fpath = pathlib.PurePath(output_dir, model_fname).as_posix()
            embedding_matrix = model.embedding.weight.data.cpu().numpy()
            torch.save(embedding_matrix, model_fpath)
            print(f'saved model into {model_fpath}')

        if batch_id >= params['n_batches']:
            break


    end_at = time.time()

    print("Elapsed time: {:.2f} [sec]".format(end_at - start_at))
    vocab.save(output_dir.as_posix())
