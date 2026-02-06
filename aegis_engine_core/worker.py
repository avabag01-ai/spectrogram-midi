import librosa

def _pyin_worker(args):
    """
    Ultra-Stable Global Worker: 
    Relocated to separate module to prevent pickling issues in macOS 'spawn' environments.
    """
    chunk, sr, hop_length = args
    return librosa.pyin(
        chunk, 
        fmin=librosa.note_to_hz('E2'), 
        fmax=librosa.note_to_hz('C6'), 
        sr=sr, 
        hop_length=hop_length
    )
