def estimate_decode_latency(num_params, bytes_per_param, bandwidth_bytes_per_s):
    latency_ms = ((num_params * bytes_per_param) / bandwidth_bytes_per_s) * 1000
    tps = 1000 / latency_ms
    return [latency_ms, tps]