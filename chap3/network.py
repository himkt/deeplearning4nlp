import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from common import word2id


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: int, 入力言語の語彙数
        :param hidden_size: int, 隠れ層のユニット数
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size,
                                      padding_idx=word2id['<PAD>'])
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, seqs, input_lengths, hidden=None):
        """
        :param seqs: tensor, 入力のバッチ, size=(max_length, batch_size)
        :param input_lengths: 入力のバッチの各サンプルの文長
        :param hidden: tensor, 隠れ状態の初期値, Noneの場合は0で初期化される
        :return output: tensor, Encoderの出力, size=(max_length, batch_size, hidden_size)
        :return hidden: tensor, Encoderの隠れ状態, size=(1, batch_size, hidden_size)
        """
        emb = self.embedding(seqs)
        packed = pack_padded_sequence(emb, input_lengths)
        output, hidden = self.gru(packed, hidden)
        output, _ = pad_packed_sequence(output)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        """
        :param hidden_size: int, 隠れ層のユニット数
        :param output_size: int, 出力言語の語彙数
        :param dropout: float, ドロップアウト率
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size,
                                      padding_idx=word2id['<PAD>'])
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, seqs, hidden):
        """
        :param seqs: tensor, 入力のバッチ, size=(1, batch_size)
        :param hidden: tensor, 隠れ状態の初期値, Noneの場合は0で初期化される
        :return output: tensor, Decoderの出力, size=(1, batch_size, output_size)
        :return hidden: tensor, Decoderの隠れ状態, size=(1, batch_size, hidden_size)
        """
        emb = self.embedding(seqs)
        output, hidden = self.gru(emb, hidden)
        output = self.out(output)
        return output, hidden


class EncoderDecoder(nn.Module):
    """EncoderとDecoderの処理をまとめる"""

    def __init__(self, input_size, output_size, hidden_size, device='cpu'):
        """
        :param input_size: int, 入力言語の語彙数
        :param output_size: int, 出力言語の語彙数
        :param hidden_size: int, 隠れ層のユニット数
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)
        self.device = device

    def forward(self, batch_X, lengths_X, max_length,
                batch_Y=None, use_teacher_forcing=False):
        """
        :param batch_X: tensor, 入力系列のバッチ, size=(max_length, batch_size)
        :param lengths_X: list, 入力系列のバッチ内の各サンプルの文長
        :param max_length: int, Decoderの最大文長
        :param batch_Y: tensor, Decoderで用いるターゲット系列
        :param use_teacher_forcing: Decoderでターゲット系列を入力とするフラグ
        :return decoder_outputs: tensor, Decoderの出力,
            size=(max_length, batch_size, self.decoder.output_size)
        """
        # encoderに系列を入力（複数時刻をまとめて処理）
        _, encoder_hidden = self.encoder(batch_X, lengths_X)

        _batch_size = batch_X.size(1)

        # decoderの入力と隠れ層の初期状態を定義
        decoder_input = torch.tensor(
            [1] * _batch_size, dtype=torch.long, device=self.device)
        # NOTE 1 is BOS
        decoder_input = decoder_input.unsqueeze(0)  # (1, batch_size)
        decoder_hidden = encoder_hidden  # Encoderの最終隠れ状態を取得

        # decoderの出力のホルダーを定義
        decoder_outputs = torch.zeros(max_length, _batch_size,
                                      self.decoder.output_size, device=self.device)

        # 各時刻ごとに処理
        for t in range(max_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden)
            decoder_outputs[t] = decoder_output
            # 次の時刻のdecoderの入力を決定
            if use_teacher_forcing and batch_Y is not None:  # ターゲット系列を用いる
                decoder_input = batch_Y[t].unsqueeze(0)
            else:  # 自身の出力を用いる
                decoder_input = decoder_output.max(-1)[1]

        return decoder_outputs


class BeamEncoderDecoder(EncoderDecoder):
    """
    Beam Searchでdecodeを行うためのクラス
    """

    def __init__(self, input_size, output_size, hidden_size, beam_size=4, device='cpu'):
        """
        :param input_size: int, 入力言語の語彙数
        :param output_size: int, 出力言語の語彙数
        :param hidden_size: int, 隠れ層のユニット数
        :param beam_size: int, ビーム数
        """
        super(BeamEncoderDecoder, self).__init__(
            input_size, output_size, hidden_size, device)
        self.beam_size = beam_size

    def forward(self, batch_X, lengths_X, max_length, **kwargs):
        """
        :param batch_X: tensor, 入力系列のバッチ, size=(max_length, batch_size)
        :param lengths_X: list, 入力系列のバッチ内の各サンプルの文長
        :param max_length: int, Decoderの最大文長
        :return decoder_outputs: list, 各ビームのDecoderの出力
        :return finished_scores: list of float, 各ビームのスコア
        """
        _, encoder_hidden = self.encoder(batch_X, lengths_X)

        _batch_size = batch_X.size(1)

        # decoderの入力と隠れ層の初期状態を定義
        decoder_input = torch.tensor(
            [1] * _batch_size, dtype=torch.long, device=self.device)
        # NOTE 1 is BOS
        decoder_input = decoder_input.unsqueeze(0)  # (1, batch_size)
        decoder_hidden = encoder_hidden

        # beam_sizeの数だけrepeatする TODO
        decoder_input = decoder_input.expand(1, self.beam_size)
        decoder_hidden = decoder_hidden.expand(1, self.beam_size, -1).contiguous()

        k = self.beam_size
        finished_beams = []
        finished_scores = []
        # 前の時刻の各ビームの対数尤度を保持しておく
        prev_probs = torch.zeros(
            self.beam_size, 1, dtype=torch.float, device=self.device)
        output_size = self.decoder.output_size

        # 各時刻ごとに処理
        for t in range(max_length):
            # decoder_input: (1, k)
            decoder_output, decoder_hidden = self.decoder(
                decoder_input[-1:], decoder_hidden)
            # decoder_output: (1, k, output_size)
            # decoder_hidden: (1, k, hidden_size)
            decoder_output_t = decoder_output[-1]  # (k, output_size)
            log_probs = prev_probs + \
                F.log_softmax(decoder_output_t, dim=-1)  # (k, output_size)
            scores = log_probs  # 対数尤度をスコアとする

            # スコアの高いビームとその単語を取得
            flat_scores = scores.view(-1)  # (k*output_size,)
            if t == 0:
                # t=0のときは後半の同じ値の繰り返しを除外
                flat_scores = flat_scores[:output_size]
            top_vs, top_is = flat_scores.data.topk(k)
            beam_indices = top_is / output_size  # (k,)
            word_indices = top_is % output_size  # (k,)

            # ビームを更新する
            _next_beam_indices = []
            _next_word_indices = []
            for b, w in zip(beam_indices, word_indices):
                if w.item() == EOS:  # EOSに到達した場合はそのビームは更新して終了
                    k -= 1
                    beam = torch.cat(
                        [decoder_input.t()[b], w.view(1,)])  # (t+2,)
                    score = scores[b, w].item()
                    finished_beams.append(beam)
                    finished_scores.append(score)
                else:   # それ以外の場合はビームを更新
                    _next_beam_indices.append(b)
                    _next_word_indices.append(w)
            if k == 0:
                break

            # tensorｎに変換
            next_beam_indices = torch.tensor(_next_beam_indices,
                                             device=self.device)
            next_word_indices = torch.tensor(_next_word_indices,
                                             device=self.device)

            # 次の時刻のDecoderの入力を更新
            decoder_input = torch.index_select(
                decoder_input, dim=-1, index=next_beam_indices)
            decoder_input = torch.cat(
                [decoder_input, next_word_indices.unsqueeze(0)], dim=0)

            # 次の時刻のDecoderの隠れ層を更新
            decoder_hidden = torch.index_select(
                decoder_hidden, dim=1, index=next_beam_indices)

            # 各ビームの対数尤度を更新
            flat_probs = log_probs.view(-1)  # (k*output_size,)
            next_indices = (next_beam_indices + 1) * next_word_indices
            prev_probs = torch.index_select(
                flat_probs, dim=0, index=next_indices).unsqueeze(1)  # (k, 1)

        # すべてのビームが完了したらデータを整形
        decoder_outputs = [[idx.item() for idx in beam[1:-1]]
                           for beam in finished_beams]

        return decoder_outputs, finished_scores
