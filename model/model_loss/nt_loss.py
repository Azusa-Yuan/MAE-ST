import torch
import torch.nn.functional as F

def infoNCEloss(q, k, null_val=None):
    T = 0.05
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    pos_sim = torch.sum(torch.mul(q, k), dim=-1)
    neg_sim = torch.matmul(q, k.transpose(-1, -2))
    pos = torch.exp(torch.div(pos_sim, T))
    neg = torch.sum(torch.exp(torch.div(neg_sim, T)), dim=-1)
    denominator = neg + pos
    return torch.mean(-torch.log(torch.div(pos, denominator)))


def nt_xent_loss(out_1, out_2, temperature):
    """Loss used in SimCLR."""
    # https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/losses/self_supervised_learning.py
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    # Negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity :
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / neg).mean()

    return loss


def nt_xent_loss_node_level(out_1, out_2, temperature=1, null_val=None):
    out = torch.cat([out_1, out_2], dim=1)
    B, N, C = out.shape

    con = torch.mm()


def cal_batch_cl_loss(x1, x2, null_val=None):
    # x1, x2 : B, F
    # 默认使用0.5
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix_a = torch.exp(sim_matrix_a / T)
    pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
    loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
    loss_a = - torch.log(loss_a).mean()

    sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
    sim_matrix_b = torch.exp(sim_matrix_b / T)
    pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
    loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
    loss_b = - torch.log(loss_b).mean()

    loss = (loss_a + loss_b) / 2
    return loss

def nt_xent_loss_simplify(x, temperature):
    assert len(x.size()) == 2

    # Cosine similarity
    xcs = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
    xcs[torch.eye(x.size(0)).bool()] = float("-inf")

    # Ground truth labels
    target = torch.arange(8)
    target[0::2] += 1
    target[1::2] -= 1

    # Standard cross-entropy loss
    return F.cross_entropy(xcs / temperature, target, reduction="mean")
