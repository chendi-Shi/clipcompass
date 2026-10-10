import torch
from torch import nn
from torch.nn import functional as F


class Recommender(nn.Module):
    def __init__(self, text_dim, visual_dim, users, dim=32, use_visual=True):
        super().__init__()
        self.use_visual = use_visual
        self.text = nn.Sequential(nn.Linear(text_dim, dim), nn.Tanh())
        self.visual = nn.Sequential(nn.Linear(visual_dim, dim), nn.Tanh())
        self.history = nn.Linear(text_dim + visual_dim, dim)
        self.user = nn.Embedding(users, dim)
        nn.init.normal_(self.user.weight, std=.02)
        self.head = nn.Sequential(nn.Linear(dim*3, dim), nn.ReLU(), nn.Linear(dim, 2))

    def item_vectors(self, text, visual):
        t = self.text(text)
        v = self.visual(visual) if self.use_visual else torch.zeros_like(t)
        return F.normalize(t + v, dim=-1)

    def user_vectors(self, ids, histories):
        h = histories.clone()
        if not self.use_visual:
            h[:, self.text[0].in_features:] = 0
        return F.normalize(self.history(h) + self.user(ids), dim=-1)

    def logits(self, user, item):
        return self.head(torch.cat([user, item, user*item], dim=-1))


def alignment_loss(model, text, visual, available):
    """Symmetric item-level InfoNCE; missing visual rows excluded."""
    text, visual = text[available], visual[available]
    if len(text) < 2 or not model.use_visual:
        return text.sum()*0
    a = F.normalize(model.text(text), dim=-1)
    b = F.normalize(model.visual(visual), dim=-1)
    scores = a @ b.T / .15
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2


def pairwise_loss(user_vectors, item_vectors, events, generator):
    """BPR pairs drawn ONLY from positive/negative training exposures of same user."""
    uids, positive, negative = [], [], []
    for u in torch.unique(events[:, 0]).tolist():
        rows = events[events[:, 0] == u]
        pos, neg = rows[rows[:, 3] == 1, 1], rows[rows[:, 3] == 0, 1]
        # Repeated exposures with mixed labels must not become false negative pairs.
        neg = neg[~torch.isin(neg, pos)]
        if not len(pos) or not len(neg):
            continue
        count = min(len(pos), 32)
        p = pos[torch.randint(len(pos), (count,), generator=generator)]
        n = neg[torch.randint(len(neg), (count,), generator=generator)]
        uids.extend([u]*count)
        positive.extend(p.tolist())
        negative.extend(n.tolist())
    if not uids:
        return item_vectors.sum()*0
    margin = (user_vectors[uids] * (item_vectors[positive] - item_vectors[negative])).sum(-1)
    return F.softplus(-margin * 5).mean()
