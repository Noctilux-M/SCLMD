import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv

class Str_encoder(nn.Module):
    def __init__(self,n_inp,n_out):
        super(Str_encoder, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_inp = n_inp
        self.n_out = n_out
        self.gatconv = GATConv((n_inp,n_inp), n_out, num_heads=3).to(device)
        self.head_linear = nn.Linear(3*n_out,   n_out).to(device)
        self.api_proj = nn.Linear(n_inp,n_inp)
        self.file_proj = nn.Linear(n_inp, n_inp)
    def forward(self,g,emb_api,emb_file,e_tensor):
        emb_api = self.api_proj(emb_api)
        emb_file = self.file_proj(emb_file)
        gat = self.gatconv(g,(emb_api,emb_file))
        z_str = self.head_linear(gat.view([gat.shape[0],-1]))
        return  z_str





