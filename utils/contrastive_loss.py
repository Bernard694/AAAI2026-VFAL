# utils/contrastive_loss.py
import torch
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.07):
    """
    z_i: [B, D]
    z_j: [B, D]
    return: scalar loss
    """
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)

    similarity_matrix = torch.matmul(z, z.T)  # [2B, 2B]
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()

    # remove self-similarity
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    logits = similarity_matrix / temperature
    loss = -torch.log((torch.exp(logits) * labels).sum(1) / torch.exp(logits).sum(1))
    return loss.mean()
