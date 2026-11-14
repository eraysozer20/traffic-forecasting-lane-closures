import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(torch.nn.Module):
    def __init__(self,device, in_channels, nb_chev_filter, num_time_filter, T, P, num_nodes, time_strides,_1stChebNet,batch_size):
        super(Net, self).__init__()
        self.device = device
        self.first_conv = nn.Conv2d(in_channels, num_time_filter, kernel_size=[1, 1], stride=[1, 1])
        self.con = nn.Conv2d(in_channels, num_time_filter, kernel_size=[1, 1], stride=[1, 1])
        self.BlockList1 = GAT_block(device = device, in_channels = 64, nb_chev_filter = nb_chev_filter, num_time_filter =  num_time_filter, T =  T, num_nodes = num_nodes, time_strides = time_strides, _1stChebNet = _1stChebNet, batch_size = batch_size)
        self.BlockList2 = GAT_block(device = device, in_channels = 64, nb_chev_filter = nb_chev_filter, num_time_filter =  num_time_filter, T =  T, num_nodes = num_nodes, time_strides = time_strides, _1stChebNet = _1stChebNet,batch_size = batch_size)
        self.final_conv = nn.Conv2d(T,  T, kernel_size=(1, num_time_filter))
        self.lstm = nn.LSTM(12,64,2,batch_first = True)
        self.linear = nn.Linear(64, P)
    def forward(self, x):
        '''
        :param x: [B, T, N, F]
        :return: (B, Num_nodes, T_out)
        '''

        list = torch.split(x,1, dim = 3) # [B, T, N, F]
        x = self.first_conv(list[0].permute(0,3,1,2)).permute(0,2,3,1)  # [B, T, N, F] - [B,F,T,N] - [B,T,N,F]
        x = self.BlockList1(x)
        x = self.BlockList2(x)  # [B, N, F, T] -> [B, T, N, F]

        a = self.final_conv(x)[:, :, :, -1] # 16, 12, 131
        output,_ = self.lstm(a.permute(2,0,1)) # 131, 16, 64

        b = self.linear(output).permute(1,2,0)

        return b


class GAT_block(nn.Module):
    def __init__(self,device, in_channels,nb_chev_filter, num_time_filter, T, num_nodes, time_strides, _1stChebNet, batch_size):
        super(GAT_block, self).__init__()
        self.TAL = Temporal_Attention_Layer(in_channels)
        self.SAL = Spatial_Attention_Layer(in_channels)
        self.cheb_conv_SAt = cheb_conv_withSAt(device,_1stChebNet, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, num_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, num_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(num_time_filter)


    def forward(self,x):
        '''
        :param x: (Batch_size, Num_nodes, F_in, T)
        :return: (Batch_size, Num_nodes, nb_time_filter, T)
        '''

        # temporal attention
        temporal_attention = self.TAL(x) # [Batch_size,F_in, T, N] = (32, 64, 6,131)

        X, spatial_attention = self.SAL(temporal_attention) # (Batchsize,Num_nodes,Num_features,T)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(X.permute(0,3,2,1), spatial_attention)  # (b, N, F_out, T)
        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 3, 2, 1))  # (b,N,F,T)->(b,F,N,T)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual.permute(0,3,1,2)

class Temporal_Attention_Layer(nn.Module):
    def __init__(self, in_channels):
        super(Temporal_Attention_Layer, self).__init__()
        num_heads = 8
        #self.batch_size = batch_size
        self.dim_per_head = in_channels // num_heads
        self.num_heads = num_heads
        self.query = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.key = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.value = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.Droupout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        x: (B, T, N, F_in)
        return: (B, F_in, T, N)
        """
        B = x.shape[0]

        query = self.query(x.permute(0, 3, 1, 2))   # (B, 64, T, N)
        key   = self.key(x.permute(0, 3, 1, 2))     # (B, 64, T, N)
        value = self.value(x.permute(0, 3, 1, 2))   # (B, 64, T, N)

        # (B, H, D, T, N)
        query = query.view(B, self.num_heads, self.dim_per_head, query.shape[2], query.shape[3])
        key   = key.view(B, self.num_heads, self.dim_per_head, key.shape[2], key.shape[3])
        value = value.view(B, self.num_heads, self.dim_per_head, value.shape[2], value.shape[3])

        # merge batch and heads: (B*H, D, T, N)
        query = query.reshape(B * self.num_heads, self.dim_per_head, query.shape[3], query.shape[4])
        key   = key.reshape(B * self.num_heads, self.dim_per_head, key.shape[3], key.shape[4])
        value = value.reshape(B * self.num_heads, self.dim_per_head, value.shape[3], value.shape[4])

        attention = torch.matmul(query, key.permute(0, 1, 3, 2))   # (B*H, D, T, T)
        attention = attention / (self.dim_per_head ** 0.5)
        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)  # (B*H, D, T, N)

        # restore heads: (B, H, D, T, N) -> (B, H*D, T, N)
        X = X.view(B, self.num_heads, self.dim_per_head, X.shape[2], X.shape[3])
        X = X.reshape(B, self.num_heads * self.dim_per_head, X.shape[3], X.shape[4])

        X = self.Droupout(X)
        return X



class Spatial_Attention_Layer(nn.Module):
    def __init__(self, in_channels):
        super(Spatial_Attention_Layer, self).__init__()
        num_heads = 8
        #self.batch_size = batch_size
        self.in_channel = nn.Parameter(torch.FloatTensor(in_channels))
        self.dim_per_head = in_channels // num_heads
        self.num_heads = num_heads
        self.query = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.key = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.value = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.Droupout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        x: (B, F_in, T, N)
        return:
            X: (B, F_in, N, T)
            attention: (B, N, N)
        """
        B = x.shape[0]

        query = self.query(x.permute(0, 1, 3, 2))   # (B, 64, N, T)
        key   = self.key(x.permute(0, 1, 3, 2))     # (B, 64, N, T)
        value = self.value(x.permute(0, 1, 3, 2))   # (B, 64, N, T)

        # (B, H, D, N, T)
        query = query.view(B, self.num_heads, self.dim_per_head, query.shape[2], query.shape[3])
        key   = key.view(B, self.num_heads, self.dim_per_head, key.shape[2], key.shape[3])
        value = value.view(B, self.num_heads, self.dim_per_head, value.shape[2], value.shape[3])

        # merge batch and heads: (B*H, D, N, T)
        query = query.reshape(B * self.num_heads, self.dim_per_head, query.shape[3], query.shape[4])
        key   = key.reshape(B * self.num_heads, self.dim_per_head, key.shape[3], key.shape[4])
        value = value.reshape(B * self.num_heads, self.dim_per_head, value.shape[3], value.shape[4])

        attention = torch.matmul(query, key.permute(0, 1, 3, 2))   # (B*H, D, N, N)
        attention = attention / (self.dim_per_head ** 0.5)
        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)  # (B*H, D, N, T)

        # restore heads
        X = X.view(B, self.num_heads, self.dim_per_head, X.shape[2], X.shape[3])
        X = X.reshape(B, self.num_heads * self.dim_per_head, X.shape[3], X.shape[4])  # (B, 64, N, T)

        attention = attention.view(B, self.num_heads, self.dim_per_head, attention.shape[2], attention.shape[3])
        attention = attention.reshape(B, self.num_heads * self.dim_per_head, attention.shape[3], attention.shape[4])  # (B, 64, N, N)
        attention = torch.matmul(attention.permute(0, 2, 3, 1), self.in_channel)  # (B, N, N)

        X = self.Droupout(X)
        return X, attention

class cheb_conv_withSAt(nn.Module):
    '''
    1-order chebyshev graph convolution
    '''

    def __init__(self,device, _1stChebNet, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self._1stChebNet = _1stChebNet
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = device
        self.Theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x:  (batch_size, N, F_in, T)  spatial_attention = [B, N, N]
        :return: (batch_size, N, F_out, T)
        '''
        x = x.permute(0, 2, 3, 1)
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        T_k_at = self._1stChebNet.mul(spatial_attention)
        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            theta = self.Theta

            rhs = T_k_at.matmul(graph_signal) # (N, N)(b, N, F_in) = (b, N, F_in)

            output = rhs.matmul(theta)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)

def model(args, A):
    '''
    :param args.device
    :param args.in_channels
    :param args.T:
    :param args.P:
    :param args.num_nodes:
    :return:
    '''
    nb_chev_filter = 64
    model = Net(args.device, args.in_channels, nb_chev_filter, 64, args.T, args.P, args.num_nodes, 1, A ,args.batch_size)


    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model