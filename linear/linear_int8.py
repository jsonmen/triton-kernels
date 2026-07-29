import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent))

import triton
import triton.language as tl
from utils import get_cuda_config
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=get_cuda_config(),
    key=["M", "N", "K"],
)
@triton.jit
def linearint8_kernel(
    in_ptr,
    weight_ptr,
    out_ptr,
    M,
    K,
    N,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_group
    start_pid_m = group_id * GROUP_SIZE
    group_size = min(num_pid_m - start_pid_m, GROUP_SIZE)
    m_pid = start_pid_m + ((pid % num_pid_group) % group_size)
    n_pid = (pid % num_pid_group) // group_size

    offsets_im = (BLOCK_SIZE_M * m_pid + tl.arange(0, BLOCK_SIZE_M)) % M
    offsets_ik = tl.arange(0, BLOCK_SIZE_K)
    offsets_wn = (BLOCK_SIZE_N * n_pid + tl.arange(0, BLOCK_SIZE_N)) % N
    in_ptrs = in_ptr + (
        offsets_im[:, None] * stride_im + offsets_ik[None, :] * stride_ik
    )
    weight_ptrs = weight_ptr + (
        offsets_wn[None, :] * stride_wn + offsets_ik[:, None] * stride_wk
    )
    # INT32 accumulator prevents overflow during reduction
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(num_pid_k):
        in_ = tl.load(in_ptrs, mask=offsets_ik[None, :] < K - k * BLOCK_SIZE_K, other=0)
        weight = tl.load(
            weight_ptrs, mask=offsets_ik[:, None] < K - k * BLOCK_SIZE_K, other=0
        )
        # Uses INT8 Tensor Cores automatically
        accumulator = tl.dot(in_, weight, accumulator, out_dtype=tl.int32)
        in_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    offs_cm = m_pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(out_ptrs, accumulator, mask=out_mask)


def linear_int8(x: torch.Tensor, weight: torch.Tensor):
    assert x.shape[1] == weight.shape[1], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix A must be contiguous"
    M, K = x.shape
    N, K = weight.shape
    # INT8 Matmul outputs to INT32
    out = torch.empty((M, N), device=x.device, dtype=torch.int32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    linearint8_kernel[grid](
        x,
        weight,
        out,
        M,
        K,
        N,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def validate_int8(M=512, N=512, K=512):
    """Custom INT8 validation verifying exact parity between Triton and torch._int_mm."""
    x = torch.randint(-128, 127, (M, K), device=DEVICE, dtype=torch.int8)
    weight = torch.randint(-128, 127, (N, K), device=DEVICE, dtype=torch.int8)

    # torch._int_mm requires B matrix to be column-major / transposed
    weight_t = weight.t().contiguous()
    result_pt = torch._int_mm(x, weight_t)

    result_triton = linear_int8(x, weight)

    max_diff = torch.max(torch.abs(result_pt - result_triton)).item()
    print(f"INT8 Validation - Max difference: {max_diff}")

    # Integer math has zero rounding error, expecting exact match
    torch.testing.assert_close(result_pt, result_triton, atol=0, rtol=0)
    print("Validation SUCCESS: Triton output matches torch._int_mm exactly!")


configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["cuBLAS", "triton"],
        line_names=["cuBLAS (_int_mm)", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TOPS",
        plot_name="matmul-performance-int8",
        args={},
    )
]


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    inp = torch.randint(-128, 127, (M, K), device=DEVICE, dtype=torch.int8)
    weight = torch.randint(-128, 127, (N, K), device=DEVICE, dtype=torch.int8)
    weight_t = weight.t()

    quantiles = [0.5, 0.2, 0.8]
    if provider == "cuBLAS":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch._int_mm(inp, weight_t), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: linear_int8(inp, weight), quantiles=quantiles
        )

    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    validate_int8()
    benchmark.run(show_plots=False, print_data=True, save_path="./")
