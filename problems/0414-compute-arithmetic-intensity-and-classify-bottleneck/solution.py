import numpy as np
def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    arithmetic_intensity = np.round(flops / bytes_accessed, 4)
    ridge_point = np.round(peak_performance / peak_bandwidth, 4)
    bottleneck = "memory-bound" if arithmetic_intensity < ridge_point else "compute-bound"
    achieved_performance = min(peak_performance, arithmetic_intensity * peak_bandwidth)
    utilization_percent = np.round((achieved_performance / peak_performance) * 100, 4)

    return {"arithmetic_intensity": arithmetic_intensity, "ridge_point": ridge_point, "bottleneck": bottleneck, "achieved_performance":achieved_performance, "utilization_percent": utilization_percent}