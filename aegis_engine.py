import concurrent.futures
import multiprocessing
import librosa
import numpy as np
import mido
import os
from mido import Message, MidiFile, MidiTrack

# Modular Core Imports
from aegis_engine_core.stems import separate_stems
from aegis_engine_core.vision import detect_rake_patterns
from aegis_engine_core.midi_logic import get_midi_events
from aegis_engine_core.tabs import generate_tabs, export_musicxml
from aegis_engine_core.worker import _pyin_worker

class AegisEngine:
    def __init__(self, sample_rate=44100, hop_length=512, n_fft=2048):
        self.sr = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft

    def load_audio(self, file_path, start_time=0, end_time=None):
        duration = (end_time - start_time) if end_time else None
        y, _ = librosa.load(file_path, sr=self.sr, offset=start_time, duration=duration)
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return y, S_dB

    def separate_stems(self, input_wav, output_dir):
        return separate_stems(input_wav, output_dir)

    def generate_tabs(self, events):
        return generate_tabs(events)

    def export_musicxml(self, tab_data, xml_path):
        return export_musicxml(tab_data, xml_path)

    def detect_rake_patterns(self, S_dB):
        return detect_rake_patterns(S_dB, self.hop_length, self.sr, 0.6)

    def audio_to_midi(self, input_wav, output_mid, **kwargs):
        """
        AI Perception Phase (Analyze Once): Returns raw data for caching.
        """
        start_time, end_time = kwargs.get('start_time', 0), kwargs.get('end_time', None)
        # Defaulting turbo_mode to False for stability in Streamlit
        turbo_mode = kwargs.get('turbo_mode', False)
        rake_sensitivity = kwargs.get('rake_sensitivity', 0.6)
        
        y, S_dB = self.load_audio(input_wav, start_time=start_time, end_time=end_time)
        if len(y) == 0:
            return None

        rake_mask = detect_rake_patterns(S_dB, self.hop_length, self.sr, rake_sensitivity)
        
        print(f"[Aegis] 🛡️ Starting Perception Phase (Turbo: {turbo_mode})...")
        if turbo_mode:
            try:
                # ONLY attempt parallel if explicitly True
                f0, voiced_flag, voiced_probs = self._parallel_pitch_tracking(y)
            except Exception as e:
                print(f"[Aegis] ⚠️ Parallel failed ({e}), falling back to stable core.")
                f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('E2'), fmax=librosa.note_to_hz('C6'), sr=self.sr, hop_length=self.hop_length)
        else:
            # Stable Path: No multiprocessing involved
            print("[Aegis] ⚓ Using Stable Single-core Analysis.")
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('E2'), fmax=librosa.note_to_hz('C6'), sr=self.sr, hop_length=self.hop_length)
            
        f0 = np.nan_to_num(f0)
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        return {
            'rake_mask': rake_mask, 'f0': f0, 'voiced_flag': voiced_flag, 
            'voiced_probs': voiced_probs, 'rms': rms, 'y': y
        }

    def extract_events(self, raw_data, output_mid, **kwargs):
        """
        Logic Filter Layer: Fast, Real-time filtering.
        """
        rake_mask, f0, voiced_flag, voiced_probs, rms = [raw_data[k] for k in ['rake_mask', 'f0', 'voiced_flag', 'voiced_probs', 'rms']]
        min_len = min(len(rake_mask), len(f0), len(rms))
        rake_mask, f0, voiced_flag, voiced_probs, rms = [arr[:min_len] for arr in [rake_mask, f0, voiced_flag, voiced_probs, rms]]

        confidence_threshold = kwargs.get('confidence_threshold', 0.70)
        vibrato_rate = kwargs.get('vibrato_rate', 5.0)
        vibrato_depth = kwargs.get('vibrato_depth', 0.3)

        final_kwargs = kwargs.copy()
        for key in ['confidence_threshold', 'start_time', 'end_time', 'turbo_mode', 'rake_sensitivity', 'vibrato_rate', 'vibrato_depth']:
            final_kwargs.pop(key, None)

        events = get_midi_events(
            rake_mask=rake_mask, f0=f0, voiced_flag=voiced_flag, active_probs=voiced_probs,
            rms=rms, sr=self.sr, hop_length=self.hop_length, confidence_threshold=confidence_threshold, **final_kwargs
        )

        if output_mid:
            mid = MidiFile()
            track_main, track_safe = MidiTrack(), MidiTrack()
            mid.tracks.extend([track_main, track_safe])
            prog = kwargs.get('midi_program', 27)
            for t in [track_main, track_safe]: t.append(Message('program_change', program=prog, time=0))

            secs_per_frame = self.hop_length / self.sr
            ticks_per_sec = mido.second2tick(1.0, ticks_per_beat=480, tempo=500000)

            midi_events = []
            for evt in events:
                st, et = int(evt['start'] * secs_per_frame * ticks_per_sec), int(evt['end'] * secs_per_frame * ticks_per_sec)

                # 해머링 온/풀 오프: velocity 감소
                technique = evt.get('technique')
                velocity = evt['velocity']
                if technique == 'hammer_on':
                    velocity = int(velocity * 0.6)  # 어택 감소
                elif technique == 'pull_off':
                    velocity = int(velocity * 0.5)  # 어택 더 감소

                midi_events.append({'t': st, 'o': 'on', 'n': evt['note'], 'tr': evt['track'], 'v': velocity})
                midi_events.append({'t': et, 'o': 'off', 'n': evt['note'], 'tr': evt['track'], 'v': 0})

                # 벤딩 Pitch Bend 이벤트 생성
                if technique == 'bend':
                    duration_ticks = et - st
                    slope_value = evt.get('slope', 0.0)

                    # slope 기반 벤딩 목표 계산 (최대 2 semitones)
                    bend_semitones = min(2.0, abs(slope_value) * 10)
                    bend_direction = 1 if slope_value > 0 else -1
                    max_bend = int(bend_direction * (bend_semitones / 2.0) * 8191)

                    num_bend_points = 15
                    for i in range(num_bend_points):
                        progress = i / num_bend_points
                        # 가속 커브 (초반에 빠르게, 끝에 천천히)
                        curve = 1 - (1 - progress) ** 2
                        bend_value = int(max_bend * curve)
                        bend_tick = st + int(progress * duration_ticks)
                        midi_events.append({'t': bend_tick, 'o': 'pitchwheel', 'tr': evt['track'], 'pitch': bend_value})

                    # 벤딩 끝에 복귀
                    midi_events.append({'t': et, 'o': 'pitchwheel', 'tr': evt['track'], 'pitch': 0})

                # Vibrato Pitch Bend 이벤트 생성
                elif technique == 'vibrato':
                    duration_ticks = et - st
                    duration_secs = duration_ticks / ticks_per_sec
                    num_bend_points = max(10, min(20, int(duration_secs * vibrato_rate * 4)))

                    for i in range(num_bend_points):
                        phase = (i / num_bend_points) * duration_secs * vibrato_rate * 2 * np.pi
                        bend_value = int(np.sin(phase) * 8191 * vibrato_depth)
                        bend_tick = st + int((i / num_bend_points) * duration_ticks)
                        midi_events.append({'t': bend_tick, 'o': 'pitchwheel', 'tr': evt['track'], 'pitch': bend_value})

                    # 비브라토 끝나면 Pitch Bend 0으로 복귀
                    midi_events.append({'t': et, 'o': 'pitchwheel', 'tr': evt['track'], 'pitch': 0})

            midi_events.sort(key=lambda x: x['t'])

            l_main, l_safe = 0, 0
            for e in midi_events:
                track = track_main if e['tr'] == 'main' else track_safe
                last = l_main if e['tr'] == 'main' else l_safe

                if e['o'] == 'pitchwheel':
                    track.append(Message('pitchwheel', pitch=e['pitch'], time=e['t'] - last))
                else:
                    track.append(Message('note_on' if e['o'] == 'on' else 'note_off', note=e['n'], velocity=e['v'], time=e['t'] - last))

                if e['tr'] == 'main': l_main = e['t']
                else: l_safe = e['t']
            
            # Correctly handle file paths vs file-like objects (BytesIO)
            if hasattr(output_mid, 'write'):
                mid.save(file=output_mid)
            else:
                mid.save(output_mid)
            
        return events

    def _parallel_pitch_tracking(self, y):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        
        num_cores = mp.cpu_count()
        duration = len(y) / self.sr
        if duration < 5.0:
            return librosa.pyin(y, fmin=librosa.note_to_hz('E2'), fmax=librosa.note_to_hz('C6'), sr=self.sr, hop_length=self.hop_length)

        total_frames = int(np.ceil(len(y) / self.hop_length))
        frames_per_core = total_frames // num_cores
        if frames_per_core == 0: frames_per_core = total_frames
        
        worker_args = []
        for i in range(num_cores):
            sf = i * frames_per_core
            if sf >= total_frames: break
            ef = (i + 1) * frames_per_core if i < num_cores - 1 else total_frames
            sy, ey = sf * self.hop_length, ef * self.hop_length
            chunk = y[sy:ey]
            if len(chunk) > 0:
                worker_args.append((chunk, self.sr, self.hop_length))
        
        # Use a fresh context to avoid BrokenPipe/Pickle issues
        ctx = mp.get_context('spawn') if os.name == 'nt' else mp.get_context('forkserver')
        try:
            with ProcessPoolExecutor(max_workers=num_cores, mp_context=ctx) as executor:
                results = list(executor.map(_pyin_worker, worker_args))
            
            f0_list, voiced_flag_list, voiced_probs_list = zip(*results)
            return np.concatenate(f0_list), np.concatenate(voiced_flag_list), np.concatenate(voiced_probs_list)
        except Exception as e:
            print(f"[Aegis] ⚠️ Parallel failed ({e}). Falling back to single-core.")
            return librosa.pyin(y, fmin=librosa.note_to_hz('E2'), fmax=librosa.note_to_hz('C6'), sr=self.sr, hop_length=self.hop_length)
