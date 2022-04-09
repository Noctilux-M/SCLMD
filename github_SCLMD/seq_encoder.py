import torch
import torch.nn as nn
import torch.nn.functional as F

def attention_mul(out_state, attn):
    cur = out_state.mul(attn)
    res = torch.sum(cur,1)
    return res

class Seq_encoder(nn.Module):
    def __init__(self, embedsize, batch_size, hid_size,out_size):
        super(Seq_encoder, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        ## Word Encoder
        self.embed = nn.Embedding(311, embedsize)
        self.wordRNN = nn.GRU(embedsize, hid_size, bidirectional=True)
        ## Word Attention
        self.wordattn = nn.Linear(2*hid_size, 2*hid_size)
        self.attn_combine = nn.Linear(2*hid_size, 2*hid_size,bias=False)
        self.emb = nn.Linear(2*hid_size, out_size)
    def forward(self,inp):
        emb_out  = self.embed(inp)
        out_state, hid_state = self.wordRNN(emb_out)
        word_annotation = self.wordattn(out_state)
        attn = F.softmax(self.attn_combine(word_annotation),dim=1)
        sent = attention_mul(out_state, attn)
        z_seq = self.emb(sent)
        return z_seq










