import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from commons import init_weights


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_z1 = nn.Conv1d(in_channels, in_channels, 1, padding=0)
        self.norm_z1 = modules.LayerNorm(in_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        z = torch.randn_like(x)
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        z = self.conv_z1(z * x_mask)
        z = self.norm_z1(z)
        x = self.conv_1((x+z) * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 window_size,
                 sf_channels=0,
                 sf_n_layers=1,
                 sf_layer=0):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = attentions.Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                                          window_size=window_size, sf_layer=sf_layer, sf_n_layers=sf_n_layers,
                                          sf_channels=sf_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, sf=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask, sf=sf)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask

    def infer_sb(self, x, x_lengths, sf=None, sf_proportion=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder.infer_sb(x * x_mask, x_mask, sf=sf, sf_proportion=sf_proportion)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask

    def remove_weight_norm(self):
        pass


class TransformerFlowBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 filter_channels,
                 kernel_size=3,
                 n_heads=2,
                 n_layers=1,
                 p_dropout=0.,
                 gin_channels=0,
                 window_size=None,
                 mean_only=False,
                 max_len=None):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.window_size = window_size

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
        self.enc = attentions.Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, window_size=window_size,
                                      gin_channels=gin_channels, g_layer=1)
        self.max_len = max_len

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g) + h
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class PriorTransformer(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 filter_channels,
                 window_size=None,
                 kernel_size=5,
                 dilation_rate=1,
                 n_t_flows=2,
                 n_c_flows=3,
                 n_t_layers=2,
                 n_c_layers=3,
                 pt_dropout=0.,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_t_layers = n_t_layers
        self.n_c_layers = n_c_layers
        self.n_t_flows = n_t_flows
        self.n_c_flows = n_c_flows
        self.gin_channels = gin_channels
        self.gin_channels = gin_channels
        self.window_size = window_size

        self.flows = nn.ModuleList()
        for i in range(n_c_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_c_layers, gin_channels=gin_channels,
                                              mean_only=True))
            self.flows.append(modules.Flip())

        for i in range(n_t_flows):
            self.flows.append(
                TransformerFlowBlock(channels, hidden_channels, filter_channels, kernel_size, n_t_layers, window_size=window_size,
                                     p_dropout=pt_dropout, gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for l in self.flows:
            if isinstance(l, modules.ResidualCouplingLayer):
                l.remove_weight_norm()


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        self.resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(self.resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class SynthesizerTrn(nn.Module):
    def __init__(self,
                 n_vocab,
                 spec_channels,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 te_dropout,
                 te_window_size,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 dp_dilation_sizes,
                 dp_filter_channels,
                 dp_dropout,
                 pt_dropout,
                 pt_window_size,
                 pt_n_t_flows,
                 pt_n_c_flows,
                 pt_n_t_layers,
                 pt_n_c_layers,
                 n_speakers=0,
                 gin_channels=0,
                 sf_channels=0,
                 sf_layer=0,
                 sf_n_layers=0,
                 duration_predictor=1,
                 **kwargs):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.te_dropout = te_dropout
        self.te_window_size = te_window_size
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.n_speakers = n_speakers
        self.sf_channels = sf_channels
        self.sf_layer = sf_layer
        self.sf_n_layers = sf_n_layers
        self.gin_channels = gin_channels
        self.duration_predictor = duration_predictor
        self.dp_dilation_sizes = dp_dilation_sizes
        self.dp_filter_channels = dp_filter_channels
        self.dp_dropout = dp_dropout
        self.pt_dropout = pt_dropout
        self.pt_window_size = pt_window_size
        self.pt_n_t_flows = pt_n_t_flows
        self.pt_n_c_flows = pt_n_c_flows
        self.pt_n_t_layers = pt_n_t_layers
        self.pt_n_c_layers = pt_n_c_layers

        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, te_dropout,
                                 window_size=te_window_size, sf_channels=sf_channels, sf_n_layers=sf_n_layers, sf_layer=sf_layer)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                             upsample_initial_channel, upsample_kernel_sizes)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = PriorTransformer(inter_channels, hidden_channels, filter_channels, pt_window_size,
                                     n_t_flows=pt_n_t_flows, n_c_flows=pt_n_c_flows, n_t_layers=pt_n_t_layers, n_c_layers=pt_n_c_layers,
                                     kernel_size=5, gin_channels=gin_channels)

        self.dp = DurationPredictor(hidden_channels, dp_filter_channels, 3, dp_dropout, gin_channels=0)

    def forward(self, x, x_lengths, sf, noise_scale=0.5, length_scale=1., max_len=None):
        sf = sf.permute(0, 2, 1)
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, sf=sf)

        logw = self.dp(x, x_mask, g=None)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=None, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len])
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()

