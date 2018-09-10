import torch.nn as nn
import torch
import numpy


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attn_dropout=0.1):
        """
        :param d_model: int, 隠れ層の次元数
        :param attn_dropout: float, ドロップアウト率
        """
        super(ScaledDotProductAttention, self).__init__()
        self.temper = numpy.power(d_model, 0.5)  # スケーリング因子
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask):
        """
        :param q: torch.tensor, queryベクトル,
            size=(n_head*batch_size, len_q, d_model/n_head)
        :param k: torch.tensor, key,
            size=(n_head*batch_size, len_k, d_model/n_head)
        :param v: torch.tensor, valueベクトル,
            size=(n_head*batch_size, len_v, d_model/n_head)
        :param attn_mask: torch.tensor, Attentionに適用するマスク,
            size=(n_head*batch_size, len_q, len_k)
        :return output: 出力ベクトル,
            size=(n_head*batch_size, len_q, d_model/n_head)
        :return attn: Attention
            size=(n_head*batch_size, len_q, len_k)
        """
        # QとKの内積(上図右側`MatMul`)でAttentionの重みを求め、スケーリングする(上図右側`Scale`)
        # WRITE ME! Hint: torch.bmmを使う (n_head*batch_size, len_q, len_k)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        # Attentionをかけたくない部分がある場合は、その部分を負の無限大に飛ばしてSoftmaxの値が0になるようにする(上図右側`Mask(opt.)`)
        attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)  # (上図右側`SoftMax`)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # WRITE ME! attnとVの内積を計算 (上図右側`MatMul`)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        :param n_head: int, ヘッド数
        :param d_model: int, 隠れ層の次元数
        :param d_k: int, keyベクトルの次元数
        :param d_v: int, valueベクトルの次元数
        :param dropout: float, ドロップアウト率
        """
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 各ヘッドごとに異なる重みで線形変換を行うための重み
        # nn.Parameterを使うことで、Moduleのパラメータとして登録できる
        self.w_qs = nn.Parameter(torch.empty(
            [n_head, d_model, d_k], dtype=torch.float))
        self.w_ks = nn.Parameter(torch.empty(
            [n_head, d_model, d_k], dtype=torch.float))
        self.w_vs = nn.Parameter(torch.empty(
            [n_head, d_model, d_v], dtype=torch.float))
        # nn.init.xavier_normal_で重みの値を初期化
        nn.init.xavier_normal_(self.w_qs)
        nn.init.xavier_normal_(self.w_ks)
        nn.init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        # 複数ヘッド分のAttentionの結果を元のサイズに写像するための線形層
        self.proj = nn.Linear(n_head*d_v, d_model)
        # nn.init.xavier_normal_で重みの値を初期化
        nn.init.xavier_normal_(self.proj.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        """
        :param q: torch.tensor, queryベクトル,
            size=(batch_size, len_q, d_model)
        :param k: torch.tensor, key,
            size=(batch_size, len_k, d_model)
        :param v: torch.tensor, valueベクトル,
            size=(batch_size, len_v, d_model)
        :param attn_mask: torch.tensor, Attentionに適用するマスク,
            size=(batch_size, len_q, len_k)
        :return outputs: 出力ベクトル,
            size=(batch_size, len_q, d_model)
        :return attns: Attention
            size=(n_head*batch_size, len_q, len_k)

        """
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        # residual connectionのための入力
        residual = q

        batch_size, len_q, d_model = q.size()
        batch_size, len_k, d_model = k.size()
        batch_size, len_v, d_model = v.size()

        # 複数ヘッド化
        # torch.repeat または .repeatで指定したdimに沿って同じテンソルを作成 [IMPORTANT]
        q_s = q.repeat(n_head, 1, 1)  # (n_head*batch_size, len_q, d_model)
        k_s = k.repeat(n_head, 1, 1)  # (n_head*batch_size, len_k, d_model)
        v_s = v.repeat(n_head, 1, 1)  # (n_head*batch_size, len_v, d_model)
        # ヘッドごとに並列計算させるために、n_headをdim=0に、batch_sizeをdim=1に寄せる
        # (n_head, batch_size*len_q, d_model)
        q_s = q_s.view(n_head, -1, d_model)
        # (n_head, batch_size*len_k, d_model)
        k_s = k_s.view(n_head, -1, d_model)
        # (n_head, batch_size*len_v, d_model)
        v_s = v_s.view(n_head, -1, d_model)

        # 各ヘッドで線形変換を並列計算(上図左側`Linear`) Attensionへの入力
        q_s = torch.bmm(q_s, self.w_qs)  # (n_head, batch_size*len_q, d_k)
        k_s = torch.bmm(k_s, self.w_ks)  # (n_head, batch_size*len_k, d_k)
        v_s = torch.bmm(v_s, self.w_vs)  # (n_head, batch_size*len_v, d_v)
        # Attentionは各バッチ各ヘッドごとに計算させるためにbatch_sizeをdim=0に寄せる
        q_s = q_s.view(-1, len_q, d_k)   # (n_head*batch_size, len_q, d_k)
        k_s = k_s.view(-1, len_k, d_k)   # (n_head*batch_size, len_k, d_k)
        v_s = v_s.view(-1, len_v, d_v)   # (n_head*batch_size, len_v, d_v)

        # Attentionを計算(上図左側`Scaled Dot-Product Attention * h`)
        outputs, attns = self.attention(
            q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))

        # 各ヘッドの結果を連結(上図左側`Concat`)
        # torch.splitでbatch_sizeごとのn_head個のテンソルに分割
        # (batch_size, len_q, d_model) * n_head
        outputs = torch.split(outputs, batch_size, dim=0)
        # dim=-1で連結
        # (batch_size, len_q, d_model*n_head)
        outputs = torch.cat(outputs, dim=-1)

        # residual connectionのために元の大きさに写像(上図左側`Linear`)
        outputs = self.proj(outputs)  # (batch_size, len_q, d_model)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs + residual)

        return outputs, attns
