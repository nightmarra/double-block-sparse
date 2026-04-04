import torch

from dbsf import find_other
from masks import mag_prune
from masks import _mag_prune_mask
from masks import _block_mask
from masks import _get_mask_2_to_4
from masks import plot_masks


def apply_tetris_reordering(A: torch.Tensor, B: torch.Tensor, prune_fn, epsilon: float = 1e-5, max_outer_iters: int = 10, max_inner_iters: int = 100):
    device = A.device
    k = A.shape[1]
    if k != B.shape[0]:
        raise ValueError("Inner dimensions of A and B mismatched.")

    W_cat = torch.cat([A.T, B], dim=1)
    perm = torch.arange(k, device=device)

    for _ in range(max_outer_iters):
        M = prune_fn(W_cat).to(W_cat.dtype)
        swaps_made = 0
        
        for _ in range(max_inner_iters):
            # tensor contraction S = |W|M^T 
            S = torch.matmul(torch.abs(W_cat), M.T) 
            L = torch.diag(S) 
            G = L.unsqueeze(1) + L.unsqueeze(0) - S - S.T
            G.fill_diagonal_(-float('inf'))
            
            max_G_idx = torch.argmax(G)
            i = max_G_idx // k
            j = max_G_idx % k
            max_g_val = G[i, j].item()
            if max_g_val <= epsilon:
                break

            # greedy swap
            w_i = W_cat[i].clone()
            W_cat[i] = W_cat[j]
            W_cat[j] = w_i
            
            p_i = perm[i].clone()
            perm[i] = perm[j]
            perm[j] = p_i

            swaps_made += 1
            
        if swaps_made == 0:
            break
            
    A_perm = A[:, perm]
    B_perm = B[perm, :]
    return A_perm, B_perm, perm


def get_tetris_prune_wrapper(mask_type, bsp_A, bsp_B, cols_A, cols_B, alt=False):
    def prune_func(W_cat):
        A_T_hat = W_cat[:, :cols_A]
        B_hat = W_cat[:, cols_A:]
        U_A_T = torch.zeros_like(A_T_hat)
        U_B = torch.zeros_like(B_hat)
        
        def get_mask(W_h, U_h, bsp, is_alt):
            if not mask_type:
                return _mag_prune_mask(W_h + U_h, bsp)
            if mask_type == 'blocks':
                return _block_mask(W_hat=W_h, U=U_h, bsparsity=bsp, rows=2, cols=2)
            if mask_type == 'blocks_alt':
                return _block_mask(W_hat=W_h, U=U_h, bsparsity=bsp, rows=1, cols=2)
            if mask_type == '2to4':
                return _get_mask_2_to_4(W_hat=W_h, U=U_h, bsparsity=bsp, transpose=not is_alt)
            if mask_type == 'hybrid':
                if not is_alt:
                    return _get_mask_2_to_4(W_hat=W_h, U=U_h, bsparsity=bsp)
                else:
                    return _mag_prune_mask(W_h + U_h, bsp)
            return _mag_prune_mask(W_h + U_h, bsp)

        mask_A_T = get_mask(A_T_hat, U_A_T, bsp_A, is_alt=False)
        mask_B = get_mask(B_hat, U_B, bsp_B, is_alt=alt)
        return torch.cat([mask_A_T, mask_B], dim=1)
        
    return prune_func


def _factorize_tetris(W, XX, mask_type, bsp=0.25, sp=0.5, mid_dim_scale=1, iters=40, tetris_at=[7]):
    original_dtype = W.dtype
    # move to fp32 for factorization
    W = W.float()
    XX = XX.float()

    # 'for the pruning of LLMs, we found that it is better
    # to project the weight matrix multiplied
    # by input feature norm'
    # norm = torch.ones_like(norm)               # for vision models
    norm = XX.diag().sqrt() + 1e-8
    
    transpose = False
    if W.shape[0] > W.shape[1]: # > ???
        W = W.T
        transpose = True
    
    nza = int(W.shape[0]**2 * bsp*2.4)
    nzb = int(W.numel() * sp - nza)
    
    if W.shape[1] == norm.shape[0]:
        Wn = W * norm.unsqueeze(0)
    elif W.shape[0] == norm.shape[0]:
        Wn = W * norm.unsqueeze(1)
    else:
        raise ValueError(f"Norm shape {norm.shape} incompatible with W {W.shape}")
    
    k_dim = int(mid_dim_scale * W.shape[0])
    if mid_dim_scale == 1:
        A = torch.eye(W.shape[0], device=W.device, dtype=torch.float32)
        B = mag_prune(Wn, (1 - nzb/2/W.numel())) 
    else:
        A = torch.rand(size=[W.shape[0], k_dim], device=W.device, dtype=torch.float32)
        B = torch.rand(size=[k_dim, W.shape[0]], device=W.device, dtype=torch.float32)

    U_a = torch.zeros_like(A)
    U_b = torch.zeros_like(B)
    alt = mask_type in ['blocks_alt', 'hybrid', '2to4']
    mt = mask_type

    for itt in range(iters):
        mask_type = mt if itt in tetris_at else mt
        if itt in tetris_at:
            bsp_A = min(0.99, 1 - nza/A.numel())
            bsp_B = min(0.99, 1 - nzb/B.numel())
            prune_fn = get_tetris_prune_wrapper(mask_type, bsp_A, bsp_B, A.shape[0], B.shape[1], alt)
            _, _, perm = apply_tetris_reordering(A, B, prune_fn)
            print(f"iteration {itt}: applied Tetris reordering")
            
            A = A[:, perm]
            B = B[perm, :]
            U_a = torch.zeros_like(A)
            U_b = torch.zeros_like(B)
            rho_itt_offset = itt

        curr_rho_itt = itt if 'rho_itt_offset' not in locals() else itt - rho_itt_offset
        rho_start = min(1.0, curr_rho_itt / (iters-3))**3
        A, U_a, mask_a = (x.T for x in find_other(
                B.T, Wn.T, nza, A.T, U_a.T, reg=1e-2, rho_start=rho_start, mask_type=mask_type
            )
        )
        B, U_b, mask_b = find_other(
                A, Wn, nzb, B, U_b, reg=1e-2, rho_start=rho_start, mask_type=mask_type, alt=alt
             )

        try:
            # if itt == iters - 1:
            #     plot_masks(mask_a.cpu(), mask_b.cpu(), mask_type)
            pass
        except NameError:
            continue
        
     # undo normalization
    if B.shape[1] == norm.shape[0]:
        B = B / norm.unsqueeze(0)
    elif B.shape[0] == norm.shape[0]:
        B = B / norm.unsqueeze(1)
    else:
        raise ValueError(f"Norm shape {norm.shape} incompatible with B {B.shape}")

    if transpose:
        res = (B.T @ A.T).to(original_dtype)
        return res, B.T.to(original_dtype), A.T.to(original_dtype)
    else:
        res = (A @ B).to(original_dtype)
        return res, A.to(original_dtype), B.to(original_dtype)
