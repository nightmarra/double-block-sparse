import math
import time

import torch
import torch.nn as nn
import transformers
import numpy as np

from einops import rearrange
import matplotlib.pyplot as plt


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def mag_prune(W, sp=0.5):
    thres = (W).abs().flatten().sort()[0][int(W.numel() * sp)]
    mask = ((W).abs() > thres)
    return W * mask

def ent(p):
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))

# dblock_e = torch.cat((torch.ones([64, 64], device="cuda"), torch.zeros([64, 64], device="cuda")))
# dblock_o = torch.cat((torch.zeros([64, 64], device="cuda"), torch.ones([64, 64], device="cuda")))
# dblock = torch.cat((dblock_e, dblock_o), dim=1)
# for _ in range(5):
#     dblock = torch.cat((dblock, dblock), dim=1)
# for _ in range(5):
#     dblock = torch.cat((dblock, dblock), dim=0)

# possible to permute and reshape, so on every block we can just call .sum()
# einops


def _mag_prune_mask(W, sp=0.6):
    thres = (W).abs().flatten().sort()[0][int(W.numel() * sp)]
    mask = ((W).abs() > thres)
    return mask

# TODO optimizations
def _get_mask_2x2(
    W_hat: torch.Tensor, 
    U: torch.Tensor, 
    bsparsity: float
) -> torch.Tensor:
    # W_hat... a 4096*4096 matrix
    # we take 2x2 blocks and if norm is below thres, we set the whole block to 0

    # get the 2x2 blocks and calculate norms
    blocks = rearrange((W_hat+U).abs(), '(h bh) (w bw) -> h w bh bw', bh=2, bw=2)
    block_norms = blocks.norm(dim=(-2, -1), p=2)

    # expand back to 4096*4096
    expanded = block_norms.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

    # construct the mask
    # flat = expanded.abs().flatten()
    # n = flat.numel()
    # k = max(1, int(bsparsity * n))
    # thres = torch.kthvalue(flat, k).values
    thres = torch.quantile(expanded.abs().flatten(), bsparsity)
    mask = (expanded.abs() > abs(thres.item()))

    del blocks, block_norms, expanded
    return mask


def _get_mask_1x2x1(
    W_hat: torch.Tensor, 
    U: torch.Tensor, 
    bsparsity: float,
    rowblock: bool
) -> torch.Tensor:
    # W_hat... a 4096*4096 matrix
    # we take 2x2 blocks and if norm is below thres, we set the whole block to 0
    if rowblock:
        x, y = 1, 2
    else:
        x, y = 2, 1

    # get the 2x2 blocks and calculate norms
    blocks = rearrange((W_hat+U).abs(), '(h bh) (w bw) -> h w bh bw', bh=x, bw=y)
    block_norms = blocks.norm(dim=(-2, -1), p=2)

    # expand back to 4096*4096
    expanded = block_norms.repeat_interleave(x, dim=0).repeat_interleave(y, dim=1)

    # construct the mask
    thres = torch.quantile(expanded.abs().flatten(), bsparsity)
    mask = (expanded.abs() > abs(thres.item()))
    
    del blocks, block_norms, expanded
    return mask


def _get_mask_2_to_4(
        W_hat: torch.Tensor, 
        U: torch.Tensor, 
        bsparsity: float) -> torch.Tensor:
    W_U_sum = W_hat + U
    rows = W_U_sum.shape[0]
    thres = int(W_U_sum.numel() * (1 - bsparsity)) # 16% or 24% of elements

    # create groups of 4 and sort the 4 elements in each one
    W_U_sum_grouped = W_U_sum.reshape(rows, -1, 4)
    g_sorted_elements, _ = W_U_sum_grouped.abs().sort(dim=-1, descending=True)
    
    # calculate norms of [top two elements] in each group
    # then sort groups according to the result
    # (we use norm as that performed best in usual block sparsity)
    top2_sums = g_sorted_elements[:, :, :2].norm(dim=-1, p=2)
    _, best_group_ids = top2_sums.flatten().sort(descending=True)
    
    group_take_count = (thres // 4) * 2
    # each group contains 4 elements, so we divide by 4, but 2:4 will be applied,
    # so we multiply by 2 to get the desired sparsity

    # group-wise mask
    # we get locations of groups where the calculated norms are the highest
    # we keep these locations in the mask
    all_groups_count = W_U_sum_grouped.shape[1]
    g_mask = torch.zeros(rows * all_groups_count, dtype=torch.bool, device=W_U_sum.device)
    g_mask[best_group_ids[:group_take_count]] = True
    g_mask = g_mask.reshape(rows, all_groups_count)

    # apply 2:4 in selected groups -> now we get back to 16%/24% sparsity
    # we select top two elements in absolute value as they contribute the most to the norm
    _, ids = W_U_sum_grouped.abs().topk(2, dim=-1, sorted=False)
    mask = torch.zeros_like(W_U_sum_grouped, dtype=torch.bool)
    mask = mask.scatter(-1, ids, True)
    mask = mask & g_mask.unsqueeze(-1) # add dim to zero out the remaining groups
    mask = mask.reshape(rows, -1)
    
    del W_U_sum_grouped, g_sorted_elements, top2_sums, g_mask
    return mask


# inner loop of the ||W-AB||^2 minization algorithm
# ADMM is performed for m (iters) iterations
def find_other2(X, W, nnz, Z, U, print_sc=None, debug=False, reg=0, rho_start=0.03, iters=5, prune_iters=2, fixmask=None, rowblock=True):
    # Z_0 = identity
    # U_0 = zero matrix
    # X can be:
    # -> A, when we're solving for B (Z) and U_b (U)
    # -> B, when we're solving for A (Z) and U_a (U)

    # normalization with diag. reg.
    XTX = X.T.matmul(X)
    norm2 = torch.diag(XTX).sqrt() + 1e-8
    An = X / norm2
    XTX = An.T.matmul(An)
    XTX += torch.diag(torch.ones_like(XTX.diag())) * XTX.diag().mean() * reg
    
    rho = 0.43 #  penalty factor set to one
    XTW = An.T.matmul(W)
    XTX_inv = torch.inverse(XTX + torch.eye(XTX.shape[1], device=XTX.device)*rho)
    XTX_inv2 = torch.inverse(XTX + torch.eye(XTX.shape[1], device=XTX.device)*rho_start)
    
    U = U * norm2.unsqueeze(1)
    Z = Z * norm2.unsqueeze(1)
    
    W_hat = XTX_inv2.matmul(XTW + rho_start*(Z-U))
    bsparsity = min(0.99, 1 - nnz/W_hat.numel()) # 0.76 or 0.84
    
    rowblock = True
    for itt in range(iters):
        if itt < prune_iters and fixmask is None:
            if rowblock:
                mask = _get_mask_2_to_4(W_hat=W_hat, U=U, bsparsity=bsparsity)
            else:
                mask = _mag_prune_mask(W_hat+U, bsparsity)
            # mask = _get_mask_2x2_fix(norm2, W_hat=W_hat, U=U, bsparsity=bsparsity)
        if fixmask is not None:
            assert fixmask.shape == Z.shape
            mask = fixmask

        # ADMM iterations
        Z = mask * (W_hat + U)
        U = U + (W_hat - Z)    
        W_hat = XTX_inv.matmul(XTW + rho*(Z-U))

    mask = mask.cpu()
    plt.imshow(mask[:16, :16])
    
    return (Z) / norm2.unsqueeze(1), U / norm2.unsqueeze(1)
# mask * (W_hat + U_copy)

# this finds AB such that ||W-AB||^2_2 is minimized
# XX is here for LLMs only
# asp = sparsity of A
def factorizeT(W, XX, bsp=0.16, sp=0.4, iters=40, fixmask=None):
    SF = 1 # scaling factor

    if fixmask is None:
        nza = int(SF*(W.shape[0]**2) * bsp) # shape of A = W_rows * W_rows
    else:
        nza = (fixmask != 0).sum().item()
    nzb = int(SF*W.numel() * sp - nza)

    # "for the pruning of LLMs, we found that it is better
    # to project the weight matrix multiplied
    # by input feature norm" - in my case too
    norm = XX.diag().sqrt().unsqueeze(1) + 1e-8
    # norm = torch.ones_like(norm)               # for vision models
    Wn = W * norm
    
    # solve the projection problem
    # A = torch.eye(n=W.shape[0], m=int(SF*W.shape[0]), device=W.device)  # identity
    # B = torch.eye(n=int(SF*W.shape[0]), m=W.shape[0], device=W.device)  # identity

    A = torch.eye(W.shape[0], device=W.device)
    B = mag_prune(Wn, (1 - nzb/2/W.numel()))    # magnitude pruning of input

    U_a = torch.zeros_like(A)
    U_b = torch.zeros_like(B)

    # inner loop
    for itt in range(iters):
        rho_start = min(1.0, itt / (iters-3))**3 # annealing
        A, U_a = (x.T for x in find_other2(
                                   B.T, Wn.T, nza, A.T, U_a.T, reg=1e-2, debug=False, rho_start=rho_start, fixmask=fixmask, rowblock=True
                               )
                            )
        B, U_b = find_other2(
                     A, Wn, nzb, B, U_b, reg=1e-2, debug=False, rho_start=rho_start, rowblock=False
                 )
        # print(f"A_shape = {A.shape}, B_shape = {B.shape}")

    return ((A / norm).matmul(B)).T, B.T, (A / norm).T


# this finds AB such that ||W-AB||^2_2 is minimized
# XX is here for LLMs only
def factorizef(W, XX, bsp=0.16, sp=0.4, iters=40, fixmask=None):
    if W.shape[0] >= W.shape[1]: # > ???
        return factorizeT(W.T, XX, bsp, sp=sp, fixmask=fixmask)
    
    if fixmask is None:
        nza = int(W.shape[0]**2 * bsp)
    else:
        nza = (fixmask != 0).sum().item()
    nzb = int(W.numel() * sp - nza)

    # "for the pruning of LLMs, we found that it is better
    # to project the weight matrix multiplied
    # by input feature norm"
    norm = XX.diag().sqrt() + 1e-8
    # norm = torch.ones_like(norm)               # for vision models
    Wn = W * norm

    # solve the projection problem
    A = torch.eye(W.shape[0], device=W.device)  # identity
    B = mag_prune(Wn, (1 - nzb/2/W.numel()))    # magnitude pruning of input
    U_a = torch.zeros_like(A)
    U_b = torch.zeros_like(B)
    
    # inner loop
    for itt in range(iters):
        rho_start = min(1.0, itt / (iters-3))**3 # annealing
        A, U_a = (x.T for x in find_other2(
                                   B.T, Wn.T, nza, A.T, U_a.T, reg=1e-2, debug=False, rho_start=rho_start, fixmask=fixmask, rowblock=True
                               )
                            )
        B, U_b = find_other2(
                     A, Wn, nzb, B, U_b, reg=1e-2, debug=False, rho_start=rho_start, rowblock=False
                 )
        print(f"A_shape = {A.shape}, B_shape = {B.shape}")
        
    return A.matmul(B / norm), A, B / norm

# XX = X^TX
# XY = BXX^TW^T
def finalize(XXb, W, Ab, Bb):
    mask = (Ab != 0).T

    XX = Bb.matmul(XXb).matmul(Bb.T)
    XY = Bb.matmul(XXb).matmul(W.T)

    norm2 = torch.diag(XX).sqrt() + 1e-8
    XX = XX / norm2 / norm2.unsqueeze(1)
    XY = XY / norm2.unsqueeze(1)
    Ax = (Ab * norm2).T.clone()

    rho = 1
    XXinv = torch.inverse(XX + torch.eye(XX.shape[1], device=XX.device)*rho)
    U = torch.zeros_like(Ax)
    for _ in range(20):
        Z = mask * (Ax + U)
        U = U + (Ax - Z)    
        Ax = XXinv.matmul(XY + rho*(Z-U))

    Ac = Z.T / norm2
    return Ac

# this finds AB such that ||XW-XAB||^2_2 is minimized
def factorize(W, XX, sparsity, nofinal=False, fixmask=None):
    if W.shape[0] == W.shape[1]:
        asp = 0.16
    else:
        asp = 0.25
    W2, Ab, Bb = factorizef(W, XX, asp=asp, sp=1-sparsity, fixmask=fixmask)
    if nofinal:
        return W2, Ab.cpu(), Bb.cpu()
    Ac = finalize(XX, W, Ab, Bb)
    W3 = Ac.matmul(Bb)
    assert W3.shape == W.shape

    # in vision models, the gradient+eigendecomposition stuff
    # is additionally performed
    return W3, Ac.cpu(), Bb.cpu()


class DoubleSparse:
    def __init__(self, layer, nofinal=False, fixmask=None):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.nofinal = nofinal
        self.fixmask= fixmask

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            raise AttributeError("Conv not supported")
        if isinstance(self.layer, transformers.Conv1D):
            raise AttributeError("Conv not supported")
        W = W.float()
        tick = time.time()

        W2, _, _ = factorize(W, self.H, sparsity, nofinal=self.nofinal, fixmask=self.fixmask)

        torch.cuda.synchronize()
        # print('time %.2f' % (time.time() - tick))

        self.layer.weight.data = W2.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # if DEBUG:
        #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()

# start on one or more matrices
# measure for target sparsity/-ies:
    # error during mag-prune only (single sparsity)
    # error double sparse
    # error after making a 2*1 or 2*2 block, prune
# if it makes sense, connect into LLM and see



# try 2x1 in A and 1x2 in B or vice versa (4 options)
# middle dimension exploration (larger)
# optionally 2:4, and combine with larger middle dimension

# 2:4 plan:
# - A aj B maju 2:4 sparsity (50%) a mid je 2048
# - cisto mag prune s 50% sparse
# - bezny doubple sparse s 50% sparse dokopy 25 A 25 B

# plt.imshow(mask[:32,:32])
