# Baseline model for "Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction"
# Source-code referred from Social-STGCNN at https://github.com/abduallahmohamed/Social-STGCNN/tree/ebd57aaf34d84763825d05cf9d4eff738d8c96bb/model.py

import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size # seq length
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(t_kernel_size, 1), padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x) #  torch.Size([1, 160, 8, 3])
        n, kc, t, v = x.size()
        # Print the values
        # print("x :first ", x.shape)
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v) # torch.Size([1, 8, 20, 8, 3])
        # print("x :view ", x.shape)
        x = torch.einsum('nkctv,kvw->nctw', (x, A)) # torch.Size([1, 20, 8, 3])
        # print("x :einsum ", x.shape)
        return x.contiguous(), A

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layers(x)


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, use_mdn=False, stride=1, dropout=0, residual=True):
        super(st_gcn, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        # self.in_channels = in_channels
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.PReLU(),
                                 nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding,),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True),)
        self.mlp = MLP(136, 8)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels),)

        self.prelu = nn.PReLU()

    def forward(self, x, A , feature):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        feature = feature.view(1, 1, -1, 1)
        feature = feature.expand(x.size(0), x.size(1), feature.size(2), x.size(3))
        x = torch.cat([x, feature], dim=2) # torch.Size([1, 20, 136, 3])
        # print(f'test after graph: shape is:{test.shape}')
        # print(f'V after graph: shape is:{x.shape}') # torch.Size([1, 20, 8, 3])
        # print(f'A after graph: shape is:{A.shape}') # torch.Size([8, 3, 3])
        x = x.permute(0, 1, 3, 2)
        # print(f'x before MLP: shape is:{x.shape}')
        x = self.mlp(x)
        x = x.permute(0, 1, 3, 2)
        # print(f'x after MLP: shape is:{x.shape}')
        x = self.tcn(x) + res
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

    # def forward(self, x, A , feature):
    #     feature = feature.view(1, 1, -1, 1)
    #     feature = feature.expand(x.size(0), x.size(1), feature.size(2), x.size(3))
    #     x = torch.cat([x, feature], dim=2)
        
    #     res = self.residual(x)
    #     x, A = self.gcn(x, A)
    #     x = self.tcn(x) + res

    #     if not self.use_mdn:
    #         x = self.prelu(x)

    #     return x, A


class social_stgcnn(nn.Module):
    def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=2, output_feat=5, seq_len=8, pred_seq_len=12, kernel_size=3):
        super(social_stgcnn, self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

    def forward(self, v, a, feature ):
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a, feature) # add feature to input please

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        return v

