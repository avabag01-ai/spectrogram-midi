import numpy as np
import soundfile as sf
import librosa

def karplus_strong(frequency, duration, sr=44100, decay_factor=0.996):
    """
    Synthesizes a plucked string sound using the Karplus-Strong algorithm.
    """
    N = int(sr / frequency)
    # Initialize ring buffer with white noise (the pluck)
    buf = np.random.uniform(-1, 1, N)
    
    n_samples = int(sr * duration)
    output = np.zeros(n_samples)
    
    # Ring buffer pointer
    ptr = 0
    
    for i in range(n_samples):
        # Output is current buffer value
        val = buf[ptr]
        output[i] = val
        
        # Update buffer: Low-pass filter (average of two samples) * decay
        # Simple LP: 0.5 * (current + previous)
        # We need the 'next' sample in the buffer which is actually the one just processed?
        # Standard KS: y[n] = 0.5 * (y[n-N] + y[n-N-1])
        
        # Circular buffer logic
        # Current 'old' value is at ptr. The previous 'old' value was at ptr-1.
        # But we are overwriting.
        
        prev_val = buf[ptr - 1] if ptr > 0 else buf[-1]
        
        # New value
        new_val = 0.5 * (val + prev_val) * decay_factor
        
        buf[ptr] = new_val
        
        ptr = (ptr + 1) % N
        
    return output

def generate_noise_rake(duration, sr=44100):
    """
    Generates a broadband noise burst (Rake).
    """
    n_samples = int(sr * duration)
    # Pink noise or White noise
    noise = np.random.normal(0, 0.8, n_samples)
    # Envelope to make it percussive
    envelope = np.linspace(1, 0, n_samples) ** 2
    return noise * envelope

def generate_test_track():
    sr = 44100
    print(f"Generating Synthetic Electric Guitar Test Data...")
    
    # 1. Clean Note: E2 (Low E)
    print("- Synthesizing E2 Note (Trend: Stable)")
    note_e2 = karplus_strong(82.41, 1.0, sr)
    
    # 2. Clean Note: A2
    print("- Synthesizing A2 Note (Trend: Stable)")
    note_a2 = karplus_strong(110.00, 1.0, sr)
    
    # 3. THE RAKE (Noise Event)
    # 20ms burst of noise
    print("- Synthesizing RAKE Pattern (Duration: ~25ms)")
    rake = generate_noise_rake(0.025, sr) 
    
    # 4. Clean Note: D3
    print("- Synthesizing D3 Note (Trend: Stable)")
    note_d3 = karplus_strong(146.83, 1.5, sr)
    
    # Assemble Track: Silence -> E2 -> Silence -> Rake -> A2 -> Rake -> D3
    silence = np.zeros(int(0.2 * sr))
    
    track = np.concatenate([
        silence,
        note_e2,
        silence, 
        rake, # The Noise to filter
        silence[:1000], # Short gap
        note_a2,
        silence,
        rake, # Another Noise
        note_d3
    ])
    
    # Normalize
    track = track / np.max(np.abs(track)) * 0.9
    
    filename = "synthetic_guitar_test.wav"
    sf.write(filename, track, sr)
    print(f"Generated: {filename}")
    return filename

if __name__ == "__main__":
    generate_test_track()
