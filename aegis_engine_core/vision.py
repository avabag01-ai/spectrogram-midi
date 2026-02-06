import numpy as np

def detect_rake_patterns(S_dB, hop_length, sr, broadband_threshold_ratio):
    """
    Pattern Recognition: Rake Detection.
    Identifies vertical lines (broadband noise) lasting 10-30ms.
    """
    n_mels, time_steps = S_dB.shape
    is_rake = np.zeros(time_steps, dtype=bool)
    
    for t in range(time_steps):
        col = S_dB[:, t]
        col_max = np.max(col)
        if col_max < -60: 
            continue
            
        active_bins = np.sum(col > (col_max - 20)) 
        ratio = active_bins / n_mels
        
        if ratio > broadband_threshold_ratio:
            is_rake[t] = True

    ms_per_frame = (hop_length / sr) * 1000
    min_frames = int(10 / ms_per_frame)
    max_frames = int(30 / ms_per_frame)
    
    final_rake_mask = np.zeros_like(is_rake)
    start = -1
    for i in range(len(is_rake)):
        if is_rake[i] and start == -1:
            start = i
        elif not is_rake[i] and start != -1:
            duration = i - start
            if min_frames <= duration <= max_frames:
                final_rake_mask[start:i] = True 
            start = -1
            
    return final_rake_mask
