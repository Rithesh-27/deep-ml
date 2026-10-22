import numpy as np
def classify_llm_phases(num_params: int, sequence_length: int, batch_size: int, bytes_per_param: int, peak_flops: float, peak_bandwidth: float) -> dict:
    """
    Analyze prefill and decode phases of LLM inference using the Roofline Model.

    Args:
        num_params: Total number of model parameters
        sequence_length: Number of input tokens processed during prefill
        batch_size: Number of sequences processed in parallel during decode
        bytes_per_param: Memory footprint per parameter (e.g., 2 for FP16)
        peak_flops: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)

    Returns:
        Dictionary containing ridge_point and analysis dicts for 'prefill' and 'decode',
        each with total_flops, memory_bytes, arithmetic_intensity, bottleneck,
        achieved_flops, and utilization_percent.
    """
    ridge_point = np.round(peak_flops / peak_bandwidth, 4)
    prefill_dict = {}
    prefill_dict["total_flops"] = 2 * num_params * sequence_length
    prefill_dict["memory_bytes"] = num_params * bytes_per_param
    prefill_dict["arithmetic_intensity"] = np.round( prefill_dict["total_flops"] / prefill_dict["memory_bytes"], 4)
    prefill_dict["bottleneck"] = "memory-bound" if prefill_dict["arithmetic_intensity"] < ridge_point else "compute-bound"
    prefill_dict["achieved_flops"] = min(peak_flops, peak_bandwidth * prefill_dict["arithmetic_intensity"])
    prefill_dict["utilization_percent"] = np.round((prefill_dict["achieved_flops"] / peak_flops) * 100, 4)
    decode_dict = {}
    decode_dict["total_flops"] = 2 * num_params * batch_size
    decode_dict["memory_bytes"] = num_params * bytes_per_param
    decode_dict["arithmetic_intensity"] = np.round( decode_dict["total_flops"] / decode_dict["memory_bytes"], 4)
    decode_dict["bottleneck"] = "memory-bound" if decode_dict["arithmetic_intensity"] < ridge_point else "compute-bound"
    decode_dict["achieved_flops"] = min(peak_flops, peak_bandwidth * decode_dict["arithmetic_intensity"])
    decode_dict["utilization_percent"] = np.round((decode_dict["achieved_flops"] / peak_flops) * 100, 4)

    return {"ridge_point": ridge_point, "prefill": prefill_dict, "decode": decode_dict}
