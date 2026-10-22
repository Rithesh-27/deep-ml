import numpy
def attention_memory_flops(B: int, h: int, N: int, d: int, bytes_per_element: int = 2) -> dict:
    """
    Compute memory traffic and FLOPs for standard self-attention.

    Args:
        B: Batch size
        h: Number of attention heads
        N: Sequence length
        d: Head dimension
        bytes_per_element: Bytes per element (e.g., 2 for FP16, 4 for FP32)

    Returns:
        dict with keys:
            'qk_flops': int - FLOPs for Q @ K^T
            'softmax_flops': int - FLOPs for softmax
            'pv_flops': int - FLOPs for P @ V
            'total_flops': int - Total FLOPs
            'memory_bytes': int - Total memory traffic in bytes
            'arithmetic_intensity': float - FLOPs per byte, rounded to 2 decimal places
    """
    qk_flops = 2 * B * h * N * d * N
    softmax_flops = 5 * B * h * N * N
    pv_flops = 2 * B * h * N * N * d
    total_flops = qk_flops + softmax_flops + pv_flops
    memory_bytes = 4 * (B * h * N * d) + 4 * (B * h * N * N)
    memory_bytes *= bytes_per_element
    arithmetic_intensity = numpy.round(total_flops / memory_bytes, 2)
    return {"qk_flops": qk_flops, "softmax_flops": softmax_flops, "pv_flops": pv_flops, "total_flops": total_flops, "memory_bytes": memory_bytes, "arithmetic_intensity": arithmetic_intensity}