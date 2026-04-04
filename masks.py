import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.colors import ListedColormap


def mag_prune(W, sp=0.5):
    thres = (W).abs().flatten().sort()[0][int(W.numel() * sp)]
    mask = ((W).abs() > thres)
    return W * mask

def _mag_prune_mask(W, sp=0.6):
    thres = (W).abs().flatten().sort()[0][int(W.numel() * sp)]
    mask = ((W).abs() > thres)
    return mask

def _block_mask(
    W_hat: torch.Tensor,
    U: torch.Tensor,
    bsparsity: float,
    rows: int, cols: int
) -> torch.Tensor:
    h, w = W_hat.shape

    blocks = (W_hat + U).reshape(h // rows, rows, w // cols, cols)
    block_norms_sq = blocks \
                        .pow(2) \
                        .sum(dim=(1, 3))  # sqrt skipped as it has no effect

    # pick top blocks
    flat = block_norms_sq.flatten()
    k = max(1, int(bsparsity * flat.numel()))
    thres = torch.kthvalue(flat, k).values

    # expand back
    block_mask = block_norms_sq > thres
    return block_mask.repeat_interleave(rows, dim=0).repeat_interleave(cols, dim=1)

def default_block_mask(W_hat, U, bsparsity):
    return _block_mask(W_hat, U, bsparsity, 2, 2)

def _get_mask_2_to_4(
    W_hat: torch.Tensor, 
    U: torch.Tensor,
    bsparsity: float,
    transpose: bool = True
) -> torch.Tensor:

    W_U_sum = W_hat + U
    if transpose:
        W_U_sum = W_U_sum.T
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
    if transpose:
        mask = mask.T
    return mask


def plot_masks(mask_a, mask_b, mask_type):
    cmap = ListedColormap(['gray', 'lime'])
    _, axs = plt.subplots(1, 2, figsize=(8, 4))

    for ax, mask, title in zip(
        axs,
        [mask_a[:16, :16], mask_b[:16, :16]],
        ['mask_a', 'mask_b']
    ):
        _ = ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)

        ax.set_xticks(np.arange(-0.525, 16, 1), minor=True,)
        ax.set_yticks(np.arange(-0.55, 16, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

        if mask_type == '2to4':
            for x in np.arange(-0.5, 16, 4):
                ax.axvline(x=x, color='deepskyblue', linewidth=1.5)
            for y in np.arange(-0.5, 16, 1):
                ax.axhline(y=y, color='deepskyblue', linewidth=1.5)

        ax.set_title(title)

    plt.tight_layout()
    plt.show()