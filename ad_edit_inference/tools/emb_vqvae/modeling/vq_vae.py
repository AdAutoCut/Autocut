import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import torch.nn.functional as F


def get_perplexity(encoding_indices, codebook_size, dtype):
    """

    :param encoding_indices: shape (nhead, bt)
    :param codebook_size:
    :param dtype:
    :return:
    """
    encode_onehot = F.one_hot(encoding_indices, codebook_size).type(dtype)  # [nhead, bt, ncode]
    # encode_onehot = encode_onehot.view(-1, codebook_size)
    avg_probs = torch.mean(encode_onehot, dim=1)  # [num_head, ncode]
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=1))  # [num_head, ]
    return perplexity


def get_utilization(encoding_indices, codebook_size, dtype):
    encode_onehot = F.one_hot(encoding_indices, codebook_size).type(dtype)  # [nhead, bt, ncode]
    codebook_count = torch.sum(encode_onehot, dim=1)  # [nhead, ncode]
    # 统计 codebook_count 不为0的个数
    utilization = torch.sum(codebook_count > 0, dim=1) / codebook_size  # [num_head, ]
    return utilization


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, mlp_size):
        """

        :param input_dim:
        :param output_dim:
        :param mlp_size:
        """
        super().__init__()
        mlp = []
        for i in range(len(mlp_size)):
            if i == 0:
                mlp.append(nn.Linear(input_dim, mlp_size[i]))
            else:
                mlp.append(nn.Linear(mlp_size[i - 1], mlp_size[i]))
            mlp.append(nn.GELU())
            mlp.append(nn.BatchNorm1d(mlp_size[i]))
            # mlp.append(nn.LayerNorm(mlp_size[0]))
        mlp.append(nn.Linear(mlp_size[-1], output_dim))
        self.net = nn.Sequential(*mlp)

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):

    def __init__(self, input_dim,
                 codebook_dim,
                 codebook_size,
                 num_head,
                 encoder_mlp_size,
                 decoder_mlp_size):

        super().__init__()
        self.num_head = num_head
        self.encoder = []
        for i in range(self.num_head):
            self.encoder.append(MLP(input_dim, codebook_dim, encoder_mlp_size))
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = MLP(codebook_dim * num_head, input_dim, decoder_mlp_size)
        self.quantizers = []
        for i in range(self.num_head):
            vq = VectorQuantize(dim=codebook_dim,
                                codebook_size=codebook_size,
                                use_cosine_sim=True,
                                threshold_ema_dead_code=2)
            self.quantizers.append(vq)
        self.quantizers = nn.ModuleList(self.quantizers)

    def forward(self, x):
        # x: (B, D)
        enc_x_list = []
        total_commit_loss = []
        total_indices = []
        for i in range(self.num_head):
            enc_x = self.encoder[i](x)
            quant_x, indices, commit_loss = self.quantizers[i](enc_x)
            enc_x_list.append(quant_x)
            total_commit_loss.append(commit_loss)
            total_indices.append(indices)

        x = torch.cat(enc_x_list, dim=-1)
        dec_x = self.decoder(x)

        total_commit_loss = sum(total_commit_loss).mean()
        total_indices = torch.stack(total_indices, dim=-1)

        return dec_x, [total_indices, total_commit_loss]

    def get_vq(self, x):
        # x: (B, D)
        total_indices = []
        for i in range(self.num_head):
            enc_x = self.encoder[i](x)
            quant_x, indices, commit_loss = self.quantizer[i](enc_x)
            total_indices.append(indices)

        total_indices = torch.cat(total_indices, dim=-1)
        return total_indices


class RQVAE(nn.Module):

    def __init__(self, input_dim,
                 codebook_dim,
                 codebook_size,
                 num_quantizers,
                 encoder_mlp_size,
                 decoder_mlp_size,
                 shared_codebook
                 ):
        super().__init__()
        self.encoder = MLP(input_dim, codebook_dim, encoder_mlp_size)
        self.decoder = MLP(codebook_dim, input_dim, decoder_mlp_size[::-1])
        self.quantizer = ResidualVQ(
            dim=codebook_dim,
            num_quantizers=num_quantizers,  # specify number of quantizers
            codebook_size=codebook_size,  # codebook size
            kmeans_init=True,  # set to True
            kmeans_iters=10,
            use_cosine_sim=True,
            threshold_ema_dead_code=2,
            shared_codebook=shared_codebook,
            ema_update=not shared_codebook
            # sample_codebook_temp=0.1,
        )

    def forward(self, x, sample_codebook_temp=0.0):
        # x: (B, D)

        # import pdb; pdb.set_trace()
        enc_x = self.encoder(x)  # (bt, input_dim)
        quant_x, indices, commit_loss = self.quantizer(enc_x, sample_codebook_temp=sample_codebook_temp)
        dec_x = self.decoder(quant_x)
        commit_loss = commit_loss.mean()

        return dec_x, [indices, commit_loss]

    def get_vq(self, x):
        # x: (B, D)
        enc_x = self.encoder(x)
        quant_x, indices, commit_loss = self.quantizer(enc_x)
        return indices

    
    def decode(self,tokens):
        pass


class MheadVQVAE(nn.Module):

    def __init__(self, input_dim,
                 codebook_dim,
                 codebook_size,
                 num_head,
                 encoder_mlp_size,
                 decoder_mlp_size):
        super().__init__()
        self.num_head = num_head
        self.encoder = MLP(input_dim, codebook_dim, encoder_mlp_size)
        self.decoder = MLP(codebook_dim, input_dim, decoder_mlp_size)
        self.quantizer = VectorQuantize(
            dim=codebook_dim,
            codebook_dim=codebook_dim // num_head,
            heads=num_head,  # specify number of quantizers
            codebook_size=codebook_size,  # codebook size
            separate_codebook_per_head=True,
            kmeans_init=True,  # set to True
            kmeans_iters=10,
            threshold_ema_dead_code=2,
        )

    def forward(self, x):
        # x: (B, D)
        enc_x = self.encoder(x)
        quant_x, indices, commit_loss = self.quantizer(enc_x)
        dec_x = self.decoder(quant_x)
        commit_loss = commit_loss.mean()

        return dec_x, [indices, commit_loss]

    def get_vq(self, x):
        # x: (B, D)
        enc_x = self.encoder(x)
        quant_x, indices, commit_loss = self.quantizer(enc_x)
        return indices
