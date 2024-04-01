import torch.nn as nn
import numpy as np
import torch
from torchinfo import summary
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, \
    Summer

def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # 为邻接矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix

class SP_PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self):
        super().__init__()

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        tp_enc_2d = PositionalEncoding2D(num_feat).to(input_data.device)
        input_data += tp_enc_2d(input_data)
        return input_data


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class TemporalAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super(TemporalAttentionLayer, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.MLP = nn.Linear(model_dim, model_dim // 2)
        self.conv1 = nn.Conv2d(model_dim, model_dim // 4, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(model_dim, model_dim // 4, kernel_size=(3, 1), padding=(1, 0))

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size, num_nodes, tgt_length, features = query.shape
        x = query
        x_reshaped = x.reshape(batch_size * num_nodes, features, tgt_length, 1)

        conv1 = torch.relu(self.conv1(x_reshaped))
        conv1 = conv1.reshape(batch_size, num_nodes, tgt_length, features // 4)
        conv2 = torch.relu(self.conv2(x_reshaped))
        conv2 = conv2.reshape(batch_size, num_nodes, tgt_length, features // 4)

        x = self.MLP(x)

        x = torch.cat((x, conv1, conv2), dim=-1)

        query = self.FC_Q(x)
        key = self.FC_K(x)
        value = self.FC_V(x)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)
        return out


class GatedSpatialAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads, adj_mx, mask=False):
        super(GatedSpatialAttentionLayer, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.adj_mx = adj_mx
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q2 = nn.Linear(model_dim, model_dim)
        self.FC_K2 = nn.Linear(model_dim, model_dim)
        self.FC_V2 = nn.Linear(model_dim, model_dim)

        self.Theta = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(0.1)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size, tgt_length, num_nodes, features = query.shape
        norm_adjmx= torch.from_numpy(norm_Adj(self.adj_mx)).type(torch.FloatTensor).to(query.device)
        Y = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0).reshape(
            batch_size * self.num_heads * tgt_length, num_nodes, self.head_dim)
        dyn_mx = torch.softmax(torch.relu((Y @ Y.transpose(-1, -2)) / (features ** 0.5)), dim=-1)
        E_g = nn.init.xavier_uniform_(nn.Parameter(torch.rand(num_nodes, num_nodes))).to(Y.device)
        dyn_mx = (dyn_mx @ E_g).reshape(batch_size * self.num_heads, tgt_length, num_nodes, num_nodes)

        out1 = torch.relu(self.Theta(torch.matmul(norm_adjmx, query))).reshape(batch_size, tgt_length, num_nodes, features)
        query2 = self.FC_Q2(query)
        key2 = self.FC_K2(key)
        value2 = self.FC_V2(value)


        query2 = torch.cat(torch.split(query2, self.head_dim, dim=-1), dim=0)
        key2 = torch.cat(torch.split(key2, self.head_dim, dim=-1), dim=0)
        value2 = torch.cat(torch.split(value2, self.head_dim, dim=-1), dim=0)


        key2 = key2.transpose(-1, -2)
        atten_score2 = (query2 @ key2) / (self.head_dim ** 0.5)
        masked_atten_score2 = torch.softmax(atten_score2.mul(dyn_mx), dim=-1)
        out2 = torch.tanh(masked_atten_score2 @ value2)
        out2 = torch.cat(torch.split(out2, batch_size, dim=0), dim=-1)

        hyper_a = 0.5
        result = hyper_a * out1 + (1 - hyper_a) * out2

        return result


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, adj_mx, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()
        self.Tattn = TemporalAttentionLayer(model_dim, num_heads, mask)
        self.Sattn = GatedSpatialAttentionLayer(model_dim, num_heads, adj_mx, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ln3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.Tattn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = out.transpose(dim, -2)
        out = self.Sattn(out, out, out)
        out = self.dropout2(out)
        out = out.transpose(dim, -2)
        out = self.ln2(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout3(out)
        out = self.ln3(residual + out)

        out = out.transpose(dim, -2)
        return out


class STTFSDP(nn.Module):
    def __init__(
            self,
            num_nodes,
            adj_mx,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.adj_mx = adj_mx
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
        )
        self.encoding = SP_PositionalEncoding()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, adj_mx, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )


    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        x = x.transpose(1, 2)
        x = self.encoding(x)
        x = x.transpose(1, 2)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    model = STTFSDP(207, 12, 12)
    summary(model, [64, 12, 207, 3])
