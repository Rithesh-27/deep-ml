import numpy as np

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    """
    Decompose end-to-end inference latency into component stages.
    
    Args:
        stage_latencies: dict mapping stage name -> np.ndarray of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])
    
    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """
    total_requests = len(stage_latencies[list(stage_latencies)[0]])
    total_latencies = np.zeros(total_requests)
    
    for stage in stage_latencies:
        total_latencies += stage_latencies[stage]
    e2e_mean = np.round(np.mean(total_latencies),2)
    e2e_percentiles = np.round(np.percentile(total_latencies,percentiles),2)
    e2e_percentiles_dict = {}
    for percentile, val in zip(percentiles,e2e_percentiles):
        e2e_percentiles_dict[percentile] = val 
    
    stage_stats = {}
    mean_stages = []
    bottleneck_stage = ""
    bottleneck_mean = 0
    for stage in stage_latencies:
        mean = np.round(np.mean(stage_latencies[stage]),2)
        mean_stages.append(mean)
        if mean > bottleneck_mean:
            bottleneck_mean = mean
            bottleneck_stage = stage
        std = np.round(np.std(stage_latencies[stage]),2)
        stage_percentiles = np.round(np.percentile(stage_latencies[stage],percentiles),2)
        stage_percentiles_dict = {}
        for percentile, val in zip(percentiles,stage_percentiles):
            stage_percentiles_dict[percentile] = val
        stage_stats[stage] = {"mean":mean, "std":std, "percentiles":stage_percentiles_dict}

    stage_pct = {}
    for stage, stage_mean in zip(list(stage_latencies),mean_stages):
        stage_pct[stage] = np.round((stage_mean / e2e_mean) * 100,2)
    return {"e2e_mean":e2e_mean, "e2e_percentiles":e2e_percentiles_dict, "stage_stats":stage_stats, "bottleneck":bottleneck_stage, "stage_pct":stage_pct}