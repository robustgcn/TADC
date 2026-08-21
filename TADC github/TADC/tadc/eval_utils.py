import torch
import numpy as np
import scipy.sparse as sp
from tadc.utils import preprocess_graph

def evaluate_acc(pred_logits, labels, idx_eval):
    pred = pred_logits[idx_eval].argmax(dim=1)
    acc = (pred == labels[idx_eval]).float().mean().item()
    return acc

def eval_single_node_acc_with_adj(adj, features, labels, split_train, split_val, split_unlabeled, target_node, device='cpu'):

    import torch.nn.functional as F
    from rramgcn.graphreshape import SmallGCN
    X = features
    y = labels
    adj_t = torch.Tensor(preprocess_graph(adj, I=False).toarray())
    if device == 'cuda':
        adj_t = adj_t.cuda()
        X = X.cuda()
        y = y.cuda()
    model = SmallGCN(X.shape[1], 50, int(y.max()+1))
    if device == 'cuda':
        model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)
    for _ in range(50):
        model.train()
        opt.zero_grad()
        out, _ = model(X, adj_t)
        loss = F.cross_entropy(out[split_train], y[split_train])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out, _ = model(X, adj_t)
        pred = out[target_node].argmax().item()
        correct = 1.0 if pred == y[target_node].item() else 0.0
    return correct
