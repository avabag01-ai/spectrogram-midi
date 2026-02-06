import numpy as np
import librosa
import scipy.signal
from mido import Message

def detect_articulations(f0, start, end, sr, hop_length):
    """
    Detects Electric Guitar techniques: Bending, Slide, or Vibrato.
    Returns: (technique, slope_value) 튜플 또는 (None, 0.0)
    """
    if end <= start: return (None, 0.0)
    slice_f0 = f0[start:end+1]
    slice_f0 = slice_f0[slice_f0 > 0]
    if len(slice_f0) < 3: return (None, 0.0)

    # 1. Slope for Slides/Bending
    notes = librosa.hz_to_midi(slice_f0)
    slope = np.polyfit(np.arange(len(notes)), notes, 1)[0]

    # Check for Vibrato (Oscillation)
    detrended = notes - np.polyval(np.polyfit(np.arange(len(notes)), notes, 1), np.arange(len(notes)))
    vibrato_amp = np.max(detrended) - np.min(detrended)

    if vibrato_amp > 0.3: # Threshold for vibrato
        return ("vibrato", slope)
    if slope > 0.05: # Significant upward trend
        return ("bend", slope)
    if abs(slope) > 0.02: # Consistent slide
        return ("slide", slope)
    return (None, 0.0)

def get_midi_events(rake_mask, f0, voiced_flag, active_probs, rms, sr, hop_length, confidence_threshold, **kwargs):
    """
    Converts raw frame data into a list of Note Events with Vector Articulations.
    """
    noise_gate_db = kwargs.get('noise_gate_db', -40)
    sustain_ms = kwargs.get('sustain_ms', 50)
    min_note_duration_ms = kwargs.get('min_note_duration_ms', 50)
    
    # Trend Line Smoothing: Median filter to ignore harmonic glitches
    try:
        # Architect's Patch: Cast voiced_flag to float64 for librosa compatibility
        f0_smooth = librosa.util.softmask(f0, voiced_flag.astype(np.float64), margin=0.5) 
        # Better: median filter for "Trend Line"
        import scipy.signal
        f0_smooth = scipy.signal.medfilt(f0_smooth, kernel_size=3)
    except Exception as e:
        print(f"[Aegis Core] ⚠️ Pitch Smoothing failed ({e}). Using raw F0.")
        f0_smooth = f0
    
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    events = []
    current_event = None
    
    min_note_duration_frames = int((min_note_duration_ms / 1000.0) * sr / hop_length)
    sustain_frames = int((sustain_ms / 1000.0) * sr / hop_length)
    
    for t in range(len(f0_smooth)):
        freq = f0_smooth[t]
        is_voiced = voiced_flag[t]
        prob = active_probs[t]
        is_rake = rake_mask[t]
        energy = rms_db[t]
        
        if energy < noise_gate_db:
            is_voiced = False
            
        if is_voiced and freq > 0 and not is_rake:
            midi_note = int(round(librosa.hz_to_midi(freq)))
            confidence = prob
            velocity = int(np.clip((energy + 80) * 1.5, 0, 127))
            
            if current_event is None:
                current_event = {
                    'note': midi_note, 'start': t, 'end': t,
                    'confidence': confidence, 'velocity': velocity,
                    'track': 'main' if confidence >= confidence_threshold else 'safe',
                    'rms_energy': energy
                }
            else:
                if current_event['note'] == midi_note:
                    current_event['end'] = t
                else:
                    technique, slope = detect_articulations(f0_smooth, current_event['start'], current_event['end'], sr, hop_length)
                    current_event['technique'] = technique
                    current_event['slope'] = slope
                    events.append(current_event)
                    current_event = {
                        'note': midi_note, 'start': t, 'end': t,
                        'confidence': confidence, 'velocity': velocity,
                        'track': 'main' if confidence >= confidence_threshold else 'safe',
                        'rms_energy': energy
                    }
        else:
            if current_event is not None:
                technique, slope = detect_articulations(f0_smooth, current_event['start'], current_event['end'], sr, hop_length)
                current_event['technique'] = technique
                current_event['slope'] = slope
                events.append(current_event)
                current_event = None

    if current_event is not None:
        technique, slope = detect_articulations(f0_smooth, current_event['start'], current_event['end'], sr, hop_length)
        current_event['technique'] = technique
        current_event['slope'] = slope
        events.append(current_event)
        
    if not events: return []
    events = [e for e in events if (e['end'] - e['start']) >= min_note_duration_frames]

    # Merge dots only if no technique detected
    if len(events) > 1:
        merged = []
        curr = events[0]
        for i in range(1, len(events)):
            next_evt = events[i]
            gap = next_evt['start'] - curr['end']
            if next_evt['note'] == curr['note'] and gap <= sustain_frames and not curr.get('technique'):
                curr['end'] = next_evt['end']
            else:
                merged.append(curr)
                curr = next_evt
        merged.append(curr)
        events = merged

    # 연속 노트 쌍 분석: 해머링 온/풀 오프 감지
    for i in range(len(events) - 1):
        curr = events[i]
        next_evt = events[i + 1]
        gap_ms = (next_evt['start'] - curr['end']) * (hop_length / sr) * 1000

        if gap_ms < 30:  # 30ms 이내 연속
            pitch_diff = next_evt['note'] - curr['note']
            # velocity와 rms_energy 모두 확인
            velocity_ratio = next_evt['velocity'] / max(curr['velocity'], 1)
            energy_ratio = next_evt.get('rms_energy', 0) / max(curr.get('rms_energy', 1), -80)

            # 해머링 온: 낮은음 → 높은음 (최대 2 semitones), 약한 어택
            if 0 < pitch_diff <= 2 and (velocity_ratio < 0.7 or energy_ratio < 0.8):
                next_evt['technique'] = 'hammer_on'
                next_evt['slope'] = 0.0  # 해머링은 slope 없음

            # 풀 오프: 높은음 → 낮은음 (최대 -2 semitones), 약한 어택
            elif -2 <= pitch_diff < 0 and (velocity_ratio < 0.7 or energy_ratio < 0.8):
                next_evt['technique'] = 'pull_off'
                next_evt['slope'] = 0.0  # 풀오프는 slope 없음

    return events
