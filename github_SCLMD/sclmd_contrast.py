import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.proj_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        self.proj_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        self.tau = tau
        self.lam = lam
        for model in self.proj_1:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.proj_2:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight)
        self.semi_classifier = nn.Linear(hidden_dim, 8)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)  # 求z1的-1范数
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos, z_semi, semi_labels):
        z_proj_mp = self.proj_1(z_mp)
        z_proj_sc = self.proj_2(z_sc)
        semi_pred = self.semi_classifier(z_semi)
        semi_loss = F.cross_entropy(semi_pred, semi_labels).requires_grad_(True)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        pos = pos + 1e-8
        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        cur1 = matrix_mp2sc.mul(pos).sum(dim=-1)
        lori_mp = -torch.log(cur1).mean()
        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        cur2 = matrix_sc2mp.mul(pos.t()).sum(dim=-1)
        lori_sc = -torch.log(cur2).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc + semi_loss








