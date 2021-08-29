import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class ClassificationTransformer(nn.Module):
    """Reference: https://github.com/649453932/Chinese-Text-Classification-Pytorch
    """
    def __init__(self, 
                 embedding_size: int,
                 last_hidden: int,
                 pad_size: int,
                 num_classes: int,
                 device: torch.device,
                 activation: nn.Module,
                 dim_model: int,
                 dropout: float=0.5,
                 num_head: int=1,
                 num_encoder: int=1,
                 hidden: int=512,
                ):
        super(ClassificationTransformer, self).__init__()
        self.activation = activation
        self.postion_embedding = Positional_Encoding(embedding_size, pad_size, dropout, device)
        self.encoder = Encoder(dim_model, num_head, hidden, dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(num_encoder)])
        
        self.dropout_layer = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(pad_size * dim_model, last_hidden)
        self.fc2 = nn.Linear(last_hidden, num_classes)


    def transformer_encoder(self, x):
        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        return out
    
    def forward(self, x):
        out = self.transformer_encoder(x)
        out = self.dropout_layer(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
    def get_encoder_vector(self, x):
        out = self.transformer_encoder(x)
        out = self.dropout_layer(out)
        return self.fc1(out)


class ClassificationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, encoder_size, class_num, device):
        super(ClassificationLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, 
            batch_first=True)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(hidden_size, encoder_size)
        self.fc2 = nn.Linear(encoder_size, class_num)
    
    def init_state(self, batch_size):
        return (torch.randn((1, batch_size, self.hidden_size), dtype=torch.float32, device=self.device, requires_grad=True), 
            torch.randn((1, batch_size, self.hidden_size), dtype=torch.float32, device=self.device, requires_grad=True))
    
    def forward(self, x):
        self.hidden_state = self.init_state(x.shape[0])
        ouputs, (ht, ct) = self.lstm(x, self.hidden_state)
        out = self.dropout_layer(ht[0])
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
    def get_encoder_vector(self, x):
        self.hidden_state = self.init_state(x.shape[0])
        ouputs, (ht, ct) = self.lstm(x, self.hidden_state)
        out = self.dropout_layer(ht[0])
        return self.fc1(out)


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention(Q, K, V)
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
