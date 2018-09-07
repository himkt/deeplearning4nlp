import pathlib
import numpy


PAD = 0
UNK = 1
PAD_TOKEN = '<PAD>'  # paddingに使います
UNK_TOKEN = '<UNK>'  # 辞書にない単語


class Vocab(object):
    def __init__(self):
        """
        word2id: 単語(str)をインデックス(int)に変換する辞書
        id2word: インデックス(int)を単語(str)に変換する辞書
        """
        self.word2id = {}
        self.word2id[PAD_TOKEN] = PAD
        self.word2id[UNK_TOKEN] = UNK
        self.id2word = {v: k for k, v in self.word2id.items()}

    def build_vocab(self, sentences, min_count=3):
        # 各単語の出現回数の辞書を作成する
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                word_counter[word] = word_counter.get(word, 0) + 1

        # min_count回以上出現する単語のみ語彙に加える
        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word

        # 語彙に含まれる単語の出現回数を保持する
        self.raw_vocab = {w: word_counter[w]
                          for w in self.word2id.keys() if w in word_counter}

    def save(self, output_dir):
        output_fpath = pathlib.PurePath(output_dir, 'vocab.npz')
        numpy.savez(output_fpath,
                    word2id=self.word2id,
                    id2word=self.id2word)

    def load(self, in_file):
        data_npz = numpy.load(in_file)
        self.word2id = data_npz['word2id'].item()
        self.id2word = data_npz['id2word'].item()
