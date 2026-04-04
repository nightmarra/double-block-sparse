import torch
from typing import Literal

from masks import mag_prune
from dbsf import factorize


@torch.no_grad()
def test_mag_prune(matrix, sp: int):
    torch.cuda.synchronize()
    
    identity = torch.eye(matrix.shape[0], device=matrix.device)
    prod = mag_prune(matrix, sp=sp)

    frobenius = torch.norm(matrix - prod, p='fro')
    print(f'Frobenius norm: {frobenius.item()}')

    prod = prod.cpu()
    print(f'AB: {prod}')
    print(f'Input matrix was: {matrix}')

    n_ab = matrix.size(dim=0)
    m_ab = matrix.size(dim=1)
    nz_count_AB = torch.count_nonzero(prod).item()

    result = float(frobenius.item())
    del identity
    torch.cuda.empty_cache()

    print(f'AB has {nz_count_AB} non-zero entries ({round(nz_count_AB/(n_ab*m_ab)*100, 1)}%)')
    return result


@torch.no_grad()
def test_double_sparse(matrix,
                       total_sp, 
                       b_bias, 
                       mid_dim_scale, 
                       mask_type: None | Literal['blocks', 'blocks_alt', '2to4', 'hybrid'] = None):
    torch.cuda.synchronize()
    
    matrix = matrix.to(dtype=torch.float32)
    identity = torch.eye(matrix.shape[0], device=matrix.device)
    prod, A, B = factorize(matrix, 
                            identity, 
                            bsp = (1-b_bias)*total_sp, 
                            sp = total_sp, 
                            mid_dim_scale = mid_dim_scale,
                            mask_type = mask_type,
                            run_finalize=True
    )

    frobenius = torch.norm(prod - matrix, p='fro')
    print(f'Frobenius norm: {frobenius.item()}')

    prod = prod.cpu()
    A = A.cpu()
    B = B.cpu()

    nz_count_A = torch.count_nonzero(A).item()
    nz_count_B = torch.count_nonzero(B).item()

    print(f'A: {A[:4, :4]}')
    print(f'B: {B[:4, :4]}')
    print(f'A.size() = {A.size()}')
    print(f'B.size() = {B.size()}')
    print(f'AB: {prod}')
    print(f'Input matrix was: {matrix}')

    n_a = A.size(dim=0)
    m_a = A.size(dim=1)
    n_b = B.size(dim=0)
    m_b = B.size(dim=1)

    del A, B, matrix, identity
    torch.cuda.empty_cache()
    
    print(f'A has {nz_count_A} non-zero entries ({round(nz_count_A/(n_a*m_a)*100, 1)}%)')
    print(f'B has {nz_count_B} non-zero entries ({round(nz_count_B/(n_b*m_b)*100, 1)}%)')

    return float(frobenius.item())


@torch.no_grad()
def test_double_block_sparse(matrix, total_sp, 
                       b_bias, 
                       mid_dim_scale):
    return test_double_sparse(matrix, total_sp, b_bias, mid_dim_scale, mask_type='blocks')

@torch.no_grad()
def test_2to4_A_B(matrix, sp=0.25, 
                  mid_dim_scale=1):
    return test_double_sparse(matrix, total_sp=2*sp, b_bias=0.5, mid_dim_scale=mid_dim_scale, mask_type='2to4')

@torch.no_grad()
def test_1x2_2x1(matrix, sp=0.25, 
                 mid_dim_scale=1):
    return test_double_sparse(matrix, total_sp=2*sp, b_bias=0.5, mid_dim_scale=mid_dim_scale, mask_type='blocks_alt')

@torch.no_grad()
def test_hybrid(matrix, asp=0.25, bsp=0.25):
    return test_double_sparse(matrix, total_sp=asp+bsp, b_bias=bsp/(asp+bsp), mid_dim_scale=1, mask_type='hybrid')
