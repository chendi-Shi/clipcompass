import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


def normalized_graph(matrix):
    users, items = matrix.shape
    graph = sp.bmat([[None, matrix], [matrix.T, None]], format="csr", dtype=np.float32)
    degree = np.asarray(graph.sum(1)).reshape(-1)
    inv = np.zeros_like(degree)
    inv[degree > 0] = degree[degree > 0] ** -.5
    graph = (sp.diags(inv) @ graph @ sp.diags(inv)).tocoo()
    return torch.sparse_coo_tensor(np.stack([graph.row, graph.col]), graph.data, (users+items, users+items), check_invariants=True).coalesce()


class GraphContent(nn.Module):
    """Independent implementation of MF / LightGCN / gated graph+content."""
    def __init__(self, users, items, content_dim, dim=32, layers=2, content=False):
        super().__init__()
        self.users, self.layers, self.use_content = users, layers, content
        self.nodes = nn.Embedding(users+items, dim)
        nn.init.normal_(self.nodes.weight, std=.1)
        if content:
            self.content = nn.Sequential(nn.Linear(content_dim, dim), nn.Tanh())
            self.history = nn.Sequential(nn.Linear(content_dim, dim), nn.Tanh())
            self.gate = nn.Linear(dim*2, 1)

    def vectors(self, graph, text, history, active_items):
        raw = self.nodes.weight
        # Never expose random/trainable ID embeddings for held-out cold items.
        raw = torch.cat([raw[:self.users], raw[self.users:]*active_items[:, None]], dim=0)
        layers = [raw]
        for _ in range(self.layers):
            layers.append(torch.sparse.mm(graph, layers[-1]))
        z = torch.stack(layers).mean(0)
        user, item = z[:self.users], z[self.users:]
        if not self.use_content:
            return user, item, None
        semantic = self.content(text)
        gate = torch.sigmoid(self.gate(torch.cat([item, semantic], dim=-1)))
        # Cold items rely entirely on content; no trained collaborative item vector.
        gate = gate * active_items[:, None]
        item = gate*item + (1-gate)*semantic
        user = user + self.history(history)
        return user, item, (z[self.users:], semantic)


def semantic_alignment(graph_items, semantic, active_items, generator, size=256):
    available = torch.nonzero(active_items, as_tuple=True)[0]
    chosen = available[torch.randperm(len(available), generator=generator)[:size]]
    if len(chosen) < 2:
        return semantic.sum()*0
    a = F.normalize(graph_items[chosen], dim=-1)
    b = F.normalize(semantic[chosen], dim=-1)
    logits = a @ b.T / .2
    labels = torch.arange(len(chosen))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))/2
