import torch.nn as nn
from str_encoder import Str_encoder
from seq_encoder import Seq_encoder
from sclmd_contrast import Contrast
import torch

class SCLMD(nn.Module):
    def __init__(self, n_inp,n_hid,n_out,batch_size,tau, lam):
        super(SCLMD, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.str_encoder = Str_encoder(n_inp,n_out).to(self.device)
        self.seq_encoder = Seq_encoder(n_inp, batch_size, n_hid, n_out).to(self.device)
        self.contrast = Contrast(n_out, tau, lam).to(self.device)

    def forward(self, pos, x_train, hetero_graph, emb_api, emb_file, semi_g, semi_labels,e_tensor):
        z_str = self.str_encoder(hetero_graph, emb_api, emb_file, e_tensor).to(self.device)
        z_seq = self.seq_encoder(x_train).to(self.device)
        open_r = len(semi_labels)
        api_num = semi_g.number_of_nodes('api')
        edge_num = semi_g.number_of_edges()
        z_semi = self.str_encoder(semi_g, emb_api[:api_num], emb_file[:open_r],e_tensor[:edge_num]).to(self.device)
        loss = self.contrast(z_str, z_seq, pos, z_semi, semi_labels)
        return loss

    def get_str_embeds(self,hetero_graph,emb_api,emb_file,e_tensor):
        z_str = self.str_encoder(hetero_graph,emb_api,emb_file,e_tensor)
        return z_str#返回str作为最终结果
