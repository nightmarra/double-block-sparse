import torch

from .masks import mag_prune
from .masks import _mag_prune_mask
from .masks import _block_mask
from .masks import _get_mask_2_to_4
from .masks import plot_masks
from .norm_perm import cluster_perm


# inner loop of the ||W-AB||_2 minization algorithm
# ADMM is performed for m (iters) iterations
def find_other(X, W, nnz, Z, U, mask_type, alt=False, reg=0, rho_start=0.03, rho=1, iters=5, prune_iters=2):
    # Z_0 = identity
    # U_0 = zero matrix
    # X can be:
    # -> A, when we're solving for B (Z) and U_b (U)
    # -> B, when we're solving for A (Z) and U_a (U)
    X, W, Z, U = X.float(), W.float(), Z.float(), U.float()

    # normalization with diag. reg.
    norm2 = torch.linalg.vector_norm(X, dim=0) + 1e-8
    An = X / norm2
    XTX = An.T.matmul(An)
    
    mean_diag = XTX.trace() / XTX.shape[0]
    if reg > 0:
        XTX.diagonal().add_(mean_diag * reg)
    
    XTW = An.T.matmul(W)

    # XTX_inv = torch.inverse(XTX + torch.eye(XTX.shape[1], device=XTX.device)*rho)
    # XTX_inv2 = torch.inverse(XTX + torch.eye(XTX.shape[1], device=XTX.device)*rho_start)
    XTX.diagonal().add_(rho_start)
    L2 = torch.linalg.cholesky(XTX)

    XTX.diagonal().add_(rho - rho_start)
    L = torch.linalg.cholesky(XTX)
    ########################

    U = U * norm2.unsqueeze(1)
    Z = Z * norm2.unsqueeze(1)
    
    # W_hat = XTX_inv2.matmul(XTW + rho_start*(Z-U))
    RHS_start = XTW + rho_start * (Z - U)
    W_hat = torch.cholesky_solve(RHS_start, L2)
    ########################

    bsparsity = min(0.99, 1 - nnz/W_hat.numel()) # 0.76 or 0.84

    for itt in range(iters):            
        if itt < prune_iters:
            if not mask_type:
                mask = _mag_prune_mask(W_hat+U, bsparsity)
            if mask_type == 'blocks':
                mask = _block_mask(W_hat=W_hat, U=U, bsparsity=bsparsity, rows=2, cols=2)

            # if mask_type == 'blocks_alt' and not alt:
            #     mask = _block_mask(W_hat=W_hat, U=U, bsparsity=bsparsity, rows=2, cols=1)
            # if mask_type == 'blocks_alt' and alt:
            #     mask = _block_mask(W_hat=W_hat, U=U, bsparsity=bsparsity, rows=1, cols=2)
            if mask_type == 'blocks_alt':
                mask = _block_mask(W_hat=W_hat, U=U, bsparsity=bsparsity, rows=1, cols=2)

            if mask_type == '2to4' and not alt:
                mask = _get_mask_2_to_4(W_hat=W_hat, U=U, bsparsity=bsparsity, transpose=True)
            if mask_type == '2to4' and alt:
                mask = _get_mask_2_to_4(W_hat=W_hat, U=U, bsparsity=bsparsity, transpose=False)

            if mask_type == 'hybrid' and not alt:
                mask = _get_mask_2_to_4(W_hat=W_hat, U=U, bsparsity=bsparsity)
            if mask_type == 'hybrid' and alt:
                mask = _mag_prune_mask(W_hat+U, bsparsity)

        # ADMM iterations
        Z = mask * (W_hat + U)
        U = U + (W_hat - Z)
        # W_hat = XTX_inv.matmul(XTW + rho*(Z-U))
        RHS = XTW + rho * (Z - U)
        W_hat = torch.cholesky_solve(RHS, L)
        ########################

    return (Z) / norm2.unsqueeze(1), U / norm2.unsqueeze(1), mask


# this finds AB such that ||W-AB||_2 is minimized
# XX is here for LLMs only
def _factorize_init(W, XX, mask_type, bsp=0.25, sp=0.5, mid_dim_scale=1, iters=40):
    original_dtype = W.dtype
    # move to fp32 for factorization
    W = W.float()
    XX = XX.float()
    
    transpose = False
    if W.shape[0] > W.shape[1]: # > ???
        W = W.T
        transpose = True
    
    nza = int(W.shape[0]*W.shape[1] * bsp)
    nzb = int(W.numel() * sp - nza)

    # 'for the pruning of LLMs, we found that it is better
    # to project the weight matrix multiplied
    # by input feature norm'
    if transpose:
        norm = XX.diag().sqrt().unsqueeze(1) + 1e-8
    else:
        norm = XX.diag().sqrt() + 1e-8
    # norm = torch.ones_like(norm)               # for vision models
    Wn = W * norm

    
    # solve the projection problem
    if mid_dim_scale == 1:
        A = torch.eye(W.shape[0], device=W.device, dtype=torch.float32)  # identity
        B = mag_prune(Wn, (1 - nzb/2/W.numel())).float()    # magnitude pruning of input
    elif mask_type == '2to4':
        A = torch.eye(n = W.shape[0], m = int(mid_dim_scale*W.shape[0]), device=W.device, dtype=torch.float32)  # rand
        B = torch.eye(n = int(mid_dim_scale*W.shape[0]), m = W.shape[0], device=W.device, dtype=torch.float32)  # rand
    else:
        A = torch.rand(size=[W.shape[0], int(mid_dim_scale*W.shape[0])], device=W.device, dtype=torch.float32)  # rand
        B = torch.rand(size=[int(mid_dim_scale*W.shape[0]), W.shape[0]], device=W.device, dtype=torch.float32)  # rand

    U_a = torch.zeros_like(A)
    U_b = torch.zeros_like(B)

    alt = False
    if mask_type in ['blocks_alt', 'hybrid', '2to4']:
        alt = True
    
    # inner loop
    for itt in range(iters):
        # print(f'iter: {itt}')
        if itt == 1 and mask_type == '2to4':
            P_matrix, _, _ = cluster_perm(A, B, Wn, iters=100)
            A = A @ P_matrix
            B = P_matrix.T @ B
            U_a = U_a @ P_matrix
            U_b = P_matrix.T @ U_b

        rho_start = min(1.0, itt / (iters-3))**3 # annealing
        A, U_a, mask_a = (x.T for x in find_other(
                   B.T, Wn.T, nza, A.T, U_a.T, reg=1e-2, rho_start=rho_start, mask_type=mask_type
                )
             )
        B, U_b, mask_b = find_other(
                A, Wn, nzb, B, U_b, reg=1e-2, rho_start=rho_start, mask_type=mask_type, alt=alt
             )
        
        # if itt == iters - 1:
        #     plot_masks(mask_a.cpu(), mask_b.cpu(), mask_type)

    if transpose:
        print(f'A.size() = {A.size()}')
        print(f'B.size() = {B.size()}')
        res_A = (A / norm).T.to(original_dtype)
        res_B = B.T.to(original_dtype)
        return res_B @ res_A, res_B, res_A
    else:
        norm = norm.unsqueeze(0)    
        res_A = A.to(original_dtype)
        res_B = (B / norm).to(original_dtype)
        return res_A @ res_B, res_A, res_B


def finalize(XX, W, A, B):
    mask = (A != 0).T
    X1 = B.matmul(XX).matmul(B.T)
    X2 = B.matmul(XX).matmul(W.T)

    norm2 = torch.diag(X1).sqrt() + 1e-8
    X1 = X1 / norm2 / norm2.unsqueeze(1)
    X2 = X2 / norm2.unsqueeze(1)
    A_temp = (A * norm2).T.clone()

    rho = 1
    X1inv = torch.inverse(X1 + torch.eye(X1.shape[1], device=X1.device)*rho)
    U = torch.zeros_like(A_temp)

    for _ in range(20):
        Z = (A_temp + U) * mask    
        U = U + (A_temp - Z)    
        A_temp = X1inv.matmul(X2 + rho*(Z - U))

    A_final = Z.T / norm2
    return A_final


def factorize(W, 
              XX, 
              mask_type, 
              bsp=0.25, sp=0.5, 
              mid_dim_scale=1, 
              run_finalize=False):
    W_temp, A_temp, B = _factorize_init(W, XX, mask_type=mask_type, bsp=bsp, sp=sp, mid_dim_scale=mid_dim_scale)
    print("Error pre-finalization: ", (W_temp - W).matmul(XX).matmul((W_temp - W).T).diag().sum().item())

    if not run_finalize:
        return W_temp, A_temp.cpu(), B.cpu()
    
    A_final = finalize(XX, W, A_temp, B)
    W_final = A_final.matmul(B)
    assert W_final.shape == W.shape
    print("Error post-finalization: ", (W_final - W).matmul(XX).matmul((W_final - W).T).diag().sum().item())
    print("Sparsity check: ", ((A_final != 0).sum() + (B != 0).sum()).item() / W.numel())
    
    return W_final, A_final.cpu(), B.cpu()
