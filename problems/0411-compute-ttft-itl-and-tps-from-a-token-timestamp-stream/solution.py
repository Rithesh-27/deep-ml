def compute_inference_metrics(timestamps: list[float]) -> dict:
	"""
	Compute LLM inference performance metrics from token timestamps.
	
	Args:
		timestamps: List of floats where timestamps[0] is the request start time
		            and timestamps[1:] are the times when each output token was generated.
	
	Returns:
		Dictionary with keys 'ttft', 'tps', 'itl' containing the metric values.
	"""
	ttft = timestamps[1] - timestamps[0]
	token_count = len(timestamps[1:])
	tps = token_count / (timestamps[-1] - timestamps[0])
	curr_time = timestamps[1]
	time_gap_sum = 0
	for time in timestamps[1:]:
		time_gap_sum += time - curr_time
		curr_time = time
	itl = time_gap_sum / max(token_count - 1,1)
	return {"ttft":ttft, "tps":tps, "itl":itl}