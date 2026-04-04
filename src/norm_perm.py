import torch


def apply_2to4_pruning(tensor, dim=-1):
    shape = tensor.shape
    if dim == 0:
        reshaped = tensor.view(-1, 4, shape[1])
        _, top_idx = reshaped.abs().topk(2, dim=1)
        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        return (reshaped * mask).view(shape)
    else:
        reshaped = tensor.view(shape[0], -1, 4)
        _, top_idx = reshaped.abs().topk(2, dim=2)
        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(2, top_idx, True)
        return (reshaped * mask).view(shape)


def evaluate_permutation_loss(A, B, W, perm_indices):
    A_perm = A[:, perm_indices]
    B_perm = B[perm_indices, :]
    
    A_pruned = apply_2to4_pruning(A_perm, dim=-1)
    B_pruned = apply_2to4_pruning(B_perm, dim=0)
    
    W_approx = A_pruned @ B_pruned
    return abs(float(torch.norm(W_approx - W, p='fro')))


def cluster_perm(A, B, W, iters=200):
    """
    Finds permutation matrix P such that APP^TB
    minimizes the 2:4 reconstruction error against W.
    """
    inner_dim = A.shape[1]
    if inner_dim % 4 != 0:
        raise ValueError("Inner dimension must be divisible by 4 for 2:4 sparsity.")
        
    importance = torch.norm(A, p=2, dim=0) * torch.norm(B, p=2, dim=1)
    sorted_idx = torch.argsort(importance, descending=True) # argsort heuristic init
    
    # interleave groups
    current_perm = torch.zeros_like(sorted_idx)
    groups = inner_dim // 4
    for i in range(inner_dim):
        group_idx = i % groups
        pos_in_group = i // groups
        current_perm[group_idx * 4 + pos_in_group] = sorted_idx[i]
    
    best_loss = evaluate_permutation_loss(A, B, W, current_perm)
    
    # greedy swap
    for i in range(iters):
        if i % 10 == 0:
            print(i)
            
        # pick two random indices to swap
        idx1, idx2 = torch.randint(0, inner_dim, (2,)).tolist()
        if idx1 == idx2: 
            continue
            
        proposed_perm = current_perm.clone()
        proposed_perm[idx1], proposed_perm[idx2] = proposed_perm[idx2], proposed_perm[idx1]
        
        loss = evaluate_permutation_loss(A, B, W, proposed_perm)
        if loss < best_loss:
            best_loss = loss
            current_perm = proposed_perm
            
    P = torch.zeros((inner_dim, inner_dim), device=A.device, dtype=A.dtype)
    P[torch.arange(inner_dim), current_perm] = 1.0
    return P, current_perm, best_loss
