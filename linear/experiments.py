import torch
import triton
from torch import nn


def to_know_what_dimenstions_and_math_to_use():
    linear = nn.Linear(28 * 28, 256)
    weights = torch.rand(256, 28 * 28)
    bias = torch.rand(256)

    x = torch.rand(64, 28 * 28)
    out = x @ weights.t() + bias
    out_torch = linear(x)
    print("x:", x.shape, x.stride())
    print("weights:", weights.shape, weights.stride())
    print("bias:", bias.shape, bias.stride())
    print("out:", out.shape, out.stride())
    print()
    print("torch weights:", linear.weight.shape, linear.weight.stride())
    print("torch bias:", linear.bias.shape, linear.bias.stride())
    print("out torch:", out_torch.shape, out_torch.stride())
    print("Are they equal?", out.allclose(out_torch))


def tiling_mm():
    batch_size = 64
    in_dim = 64
    out_dim = 64
    block_size_batch = 4
    block_size_in = 8
    block_size_out = 8
    grid_size = triton.cdiv(batch_size, block_size_batch) * triton.cdiv(
        out_dim, block_size_out
    )

    in_tensor = torch.rand(batch_size, in_dim)
    weight_tensor = torch.rand(out_dim, in_dim)
    out_tensor = torch.zeros(batch_size, out_dim)
    in_view = in_tensor.view(-1)
    weight_view = weight_tensor.view(-1)
    out_view = out_tensor.view(-1)

    m, k = in_tensor.stride()
    n, k = weight_tensor.stride()
    m_blocks = triton.cdiv(batch_size, block_size_batch)
    k_blocks = triton.cdiv(in_dim, block_size_in)
    n_blocks = triton.cdiv(out_dim, block_size_out)
    true_mm = in_tensor @ weight_tensor.T
    true_block = true_mm[0:block_size_batch, 0:block_size_out]

    for pid in range(grid_size):
        m_block_pid = pid // n_blocks
        n_block_pid = pid % n_blocks
        m_offsets = torch.arange(0, block_size_batch)[:, None]
        k_offsets = torch.arange(0, block_size_in)
        n_offsets = torch.arange(0, block_size_out)[None, :]
        acc = torch.zeros(block_size_batch, block_size_out)
        acc_true = torch.zeros(block_size_batch, block_size_out)

        for ki in range(k_blocks):
            mk_offsets = (
                m_offsets * m
                + (m_block_pid * block_size_batch * m)
                + k_offsets[None, :]
                + (ki * block_size_in)
            )
            nk_offsets = (
                n_offsets * n
                + (n_block_pid * block_size_out * n)
                + k_offsets[:, None]
                + (ki * block_size_in)
            )
            acc += in_view[mk_offsets] @ weight_view[nk_offsets]

        out_m = (m_offsets + m_block_pid * block_size_batch) * m
        out_n = n_offsets + n_block_pid * block_size_out
        out_offsets = out_m + out_n
        out_view[out_offsets] = acc
    print(torch.allclose(true_mm, out_tensor))


def mm_swizzling():
    group_size = 3
    batch_size = 7 * 8
    in_dim = 9 * 8
    out_dim = 13 * 8
    block_size_batch = 8
    block_size_in = 8
    block_size_out = 8
    grid_size = triton.cdiv(batch_size, block_size_batch) * triton.cdiv(
        out_dim, block_size_out
    )

    in_tensor = torch.rand(batch_size, in_dim)
    weight_tensor = torch.rand(out_dim, in_dim)
    out_tensor = torch.zeros(batch_size, out_dim)
    in_view = in_tensor.view(-1)
    weight_view = weight_tensor.view(-1)
    out_view = out_tensor.view(-1)

    m, k = in_tensor.stride()
    n, k = weight_tensor.stride()
    m_blocks = triton.cdiv(batch_size, block_size_batch)
    k_blocks = triton.cdiv(in_dim, block_size_in)
    n_blocks = triton.cdiv(out_dim, block_size_out)
    true_mm = in_tensor @ weight_tensor.T
    true_block = true_mm[0:block_size_batch, 0:block_size_out]
    num_pid_in_group = group_size * n_blocks
    print(f"{num_pid_in_group=}")

    for pid in range(grid_size):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_size
        group_size_m = min(m_blocks - first_pid_m, group_size)
        m_block_pid = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        n_block_pid = (pid % num_pid_in_group) // group_size_m
        print(m_block_pid, n_block_pid)

        m_offsets = torch.arange(0, block_size_batch)[:, None]
        k_offsets = torch.arange(0, block_size_in)
        n_offsets = torch.arange(0, block_size_out)[None, :]
        acc = torch.zeros(block_size_batch, block_size_out)
        acc_true = torch.zeros(block_size_batch, block_size_out)

        for ki in range(k_blocks):
            mk_offsets = (
                m_offsets * m
                + (m_block_pid * block_size_batch * m)
                + k_offsets[None, :]
                + (ki * block_size_in)
            )
            nk_offsets = (
                n_offsets * n
                + (n_block_pid * block_size_out * n)
                + k_offsets[:, None]
                + (ki * block_size_in)
            )
            acc += in_view[mk_offsets] @ weight_view[nk_offsets]

        out_m = (m_offsets + m_block_pid * block_size_batch) * m
        out_n = n_offsets + n_block_pid * block_size_out
        out_offsets = out_m + out_n
        out_view[out_offsets] = acc
