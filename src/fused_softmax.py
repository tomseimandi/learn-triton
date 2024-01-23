"""
Fused softmax: softmax operation that is significantly faster
than PyTorch's native op for a particular class of matrices:
those whose rows can fit in the GPU's SRAM.
"""
import torch
import triton
import triton.language as tl


@torch.jit.script
def naive_softmax(x):
    """
    Compute row-wise softmax of X using native Pytorch.
    """
    # Counting reads from and writes to memory (DRAM)
    # x has shape (M, N)
    numerator = torch.exp(x)
    # Step 1: MN reads, MN writes
    denominator = numerator.sum(dim=-1)
    # Step 2: MN reads, M writes
    z = numerator / denominator[:, None]
    # Step 3: MN + M reads, MN writes
    # Total, 3MN + M reads, 2MN + M writes
    return z


@torch.jit.script
def naive_softmax_2(x):
    """
    Compute row-wise softmax of X using native Pytorch.
    """
    # Counting reads from and writes to memory (DRAM)
    # x has shape (M, N)
    numerator = torch.exp(x)
    # Step 1: MN reads, MN writes
    z = numerator / numerator.sum(dim=-1)[:, None]
    # Step 2: MN reads, MN writes
    # Total, 2MN reads, 2MN writes
    return z


@torch.jit.script
def naive_softmax_no_overflow(x):
    """
    Compute row-wise softmax of X using native Pytorch.
    We substract the maximum element to avoid overflows. Softmax
    is invariant to this shift.
    """
    x_max = x.max(dim=-1)[0]
    # MN reads, M writes
    z = x - x_max[:, None]
    # MN + M reads, MN writes
    numerator = torch.exp(z)
    # MN reads, MN writes
    denominator = numerator.sum(dim=-1)
    # MN reads, M writes
    ret = numerator / denominator[:, None]
    # MN + M reads, MN writes
    # Total: 5MN + 2M reads, 3MN + 2M writes
    return ret


@triton.jit
def softmax_kernel(
    Y,
    X,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for fused softmax. Each program loads a row of the input
    matrix X, normalizes it and writes back the result to the output Y.

    Each block must have a power-of-two number of elements, so we need to internally
    "pad" each row and guard the memory operations properly if we want to handle
    any possible input shapes:

    Args:
        Y: Output pointer.
        X: Input pointer.
        input_row_stride: Input row stride.
        output_row_stride: Output row stride.
        n_cols: Number of columns.
        BLOCK_SIZE (tl.constexpr): Block size.
    """
    # Each program loads a row of the input
    row_idx = tl.program_id(axis=0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = X + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = Y + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax(x):
    """
    Helper fn for softmax.

    Args:
        x: Input tensor.
    """
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the
    # number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more
    # threads per row by increasing the number of warps (`num_warps`)
    # over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value
    # in a more natural way so you don't have to come up with manual
    # heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel
    # instance per row of the input matrix
    softmax_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


@triton.jit
def softmax_kernel_2(
    Y,
    X,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for fused softmax. Each program loads a row of the input
    matrix X, normalizes it and writes back the result to the output Y.

    Each block must have a power-of-two number of elements, so we need to internally
    "pad" each row and guard the memory operations properly if we want to handle
    any possible input shapes:

    Args:
        Y: Output pointer.
        X: Input pointer.
        n_cols: Number of columns.
        BLOCK_SIZE (tl.constexpr): Block size.
    """
    # Each program loads a row of the input
    row_idx = tl.program_id(axis=0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = X + row_idx * n_cols
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = Y + row_idx * n_cols
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax_2(x):
    """
    Helper fn for softmax.

    Args:
        x: Input tensor.
    """
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the
    # number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more
    # threads per row by increasing the number of warps (`num_warps`)
    # over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value
    # in a more natural way so you don't have to come up with manual
    # heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel
    # instance per row of the input matrix
    softmax_kernel_2[(n_rows, )](
        y,
        x,
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'triton',
            'triton2',
            'torch-native',
            'torch-jit',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Triton simplified",
            "Torch (native)",
            "Torch (jit)",
        ],  # label name for the lines
        styles=[('blue', '-'), ('blue', '--'), ('green', '-'), ('green', '--')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    if provider == 'triton2':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax_2(x), quantiles=quantiles)
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax_no_overflow(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = softmax(x)
    y_triton_2 = softmax_2(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    assert torch.allclose(y_triton_2, y_torch), (y_triton_2, y_torch)
    benchmark.run(show_plots=True, print_data=True)
