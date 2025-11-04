import math
import time

import torch
import torch.nn as nn
import transformers
import numpy as np

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def mag_prune(W, sp=0.6):
    thres = (W).abs().flatten().sort()[0][int(W.numel() * sp)]
    mask = ((W).abs() > thres)
    return W * mask

def ent(p):
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))

# inner loop of the ||W-AB||^2 minization algorithm
# ADMM is performed for m (iters) iterations
def find_other2(X, W, nnz, Z, U, print_sc=None, debug=False, reg=0, rho_start=0.03, iters=5, prune_iters=2, fixmask=None):
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
    
    rho = 1 # penalty factor set to one
    XTW = An.T.matmul(W)
    XTX_inv = torch.inverse(XTX + torch.eye(XTX.shape[1], device=XTX.device)*rho)
    XTX_inv2 = torch.inverse(XTX + torch.eye(XTX.shape[1], device=XTX.device)*rho_start)
    
    U = U * norm2.unsqueeze(1)
    Z = Z * norm2.unsqueeze(1)
    
    W_hat = XTX_inv2.matmul(XTW + rho_start*(Z-U))
    bsparsity = min(0.99, 1 - nnz/W_hat.numel())

    for itt in range(iters):
        if itt < prune_iters and fixmask is None:
            thres = (W_hat+U).abs().flatten().sort()[0][int(W_hat.numel() * bsparsity)]
            mask = ((W_hat+U).abs() > thres) # TODO target: blocks here!!!
            del thres
        if fixmask is not None:
            assert fixmask.shape == Z.shape
            mask = fixmask

        # ADMM iterations
        Z = mask * (W_hat + U)
        U = U + (W_hat - Z)    
        W_hat = XTX_inv.matmul(XTW + rho*(Z-U))

    return Z / norm2.unsqueeze(1), U / norm2.unsqueeze(1)    


# this finds AB such that ||W-AB||^2_2 is minimized
# XX is here for LLMs only
# asp = sparsity of A
def factorizeT(W, XX, asp=0.16, sp=0.4, iters=40, fixmask=None):
    if fixmask is None:
        nza = int(W.shape[0]**2 * asp) # shape of A = W_rows * W_rows
    else:
        nza = (fixmask != 0).sum().item()
    nzb = int(W.numel() * sp - nza)

    # "for the pruning of LLMs, we found that it is better
    # to project the weight matrix multiplied
    # by input feature norm" - in my case too
    norm = XX.diag().sqrt().unsqueeze(1) + 1e-8
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
        A, U_a = (x.T for x in find_other2(B.T, Wn.T, nza, A.T, U_a.T, reg=1e-2, debug=False, rho_start=rho_start, fixmask=fixmask))
        B, U_b = find_other2(A, Wn, nzb, B, U_b, reg=1e-2, debug=False, rho_start=rho_start)

    return ((A / norm).matmul(B)).T, B.T, (A / norm).T


# this finds AB such that ||W-AB||^2_2 is minimized
# XX is here for LLMs only
def factorizef(W, XX, asp=0.16, sp=0.4, iters=40, fixmask=None):
    if W.shape[0] >= W.shape[1]:
        return factorizeT(W.T, XX, asp, sp=sp, fixmask=fixmask)
    
    if fixmask is None:
        nza = int(W.shape[0]**2 * asp)
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
        A, U_a = (x.T for x in find_other2(B.T, Wn.T, nza, A.T, U_a.T, reg=1e-2, debug=False, rho_start=rho_start, fixmask=fixmask))
        B, U_b = find_other2(A, Wn, nzb, B, U_b, reg=1e-2, debug=False, rho_start=rho_start)
        
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
