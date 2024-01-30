import torch
import torch.nn as nn
import torch.nn.functional as F
from 存档和备份.STAE.STAEformer import STIF
class FGN(nn.Module):
    def __init__(self, if_STIM, pre_length, embed_size,
                 feature_size, seq_length, hidden_size, num_nodes, node_feats, hard_thresholding_fraction=1,
                 output_dim=1, hidden_size_factor=1, sparsity_threshold=0.01, input_dim=3):
        super().__init__()
        self.if_STIM = if_STIM
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.input_dim = input_dim
        # self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length * output_dim)
        )
        self.to('cuda:0')
        if self.if_STIM:
            self.STIM = self.init_embedding = STIF(3, 24, 288, 24,
                                   24, 40, 40, node_feats,
                                   12, num_nodes)
            self.tokenEmb = nn.Linear(152, self.embed_size, bias=False)
        else:
            self.tokenEmb = nn.Linear(self.input_dim, self.embed_size, bias=False)

    # def tokenEmb(self, x):
    #     x = x.unsqueeze(2)
    #     y = self.embeddings
    #     return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        o1_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        # 一种特殊的激活函数F.softshrink是PyTorch中的一个函数，用于对张量进行 soft shrinkage 操作，soft shrinkage 是一种软阈值操作。soft shrinkage 的作用是对张量中的每个元素进行阈值处理，当元素绝对值小于阈值时，将其设为零；当元素绝对值大于等于阈值时，对元素进行符号保持并减去阈值的绝对值。
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
                torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
                torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
                self.b3[0]
        )

        o3_imag = F.relu(
                torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
                torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
                self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        # B, (N*L)//2+1, self.frequency_size 2
        # torch.view_as_complex是PyTorch中的一个函数，用于将实数张量转换为复数张量。该函数会把输入张量的最后一个维度拆分成两个维度（实部和虚部），并将这两个维度合并成一个复数维度，生成一个新的形状为(..., 2)的张量。
        # 具体来说，如果输入张量的形状为(..., n)，那么torch.view_as_complex会生成一个形状为(..., n/2)的复数张量
        z = torch.view_as_complex(z)
        return z

    def forward(self, history_data: torch.Tensor, pre_train_embedding, future_data: torch.Tensor=None, batch_seen: int=0, epoch: int=0, train: bool=True, **kwargs):
        # input: B, L, N, C

        x = pre_train_embedding

        if self.if_STIM:
            x = self.STIM(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        B, N, L, C = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1, C)
        # embedding B*NL ==> B*NL*D

        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        #
        x = x.reshape(B, (N*L)//2+1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias

        x = x.reshape(B, (N*L)//2+1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x).reshape(B, N, self.pre_length, -1).permute(0, 2, 1, 3)

        return x

