import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os
import soundfile as sf
import io


def verify_technique_by_audio_matching(events, raw_data, engine, synthesizer, sr, hop_length):
    """
    감지된 테크닉(bend, hammer_on, pull_off)을 오디오 패턴 매칭으로 검증.

    프로세스:
    1. 테크닉이 감지된 이벤트의 MIDI 구간 추출
    2. 해당 구간만 FluidSynth로 합성 (테크닉 있는 버전 vs 없는 버전)
    3. 원본 오디오의 같은 구간과 비교 (spectral correlation)
    4. 유사도가 높은 버전으로 확정

    Args:
        events: 이벤트 리스트 (technique 포함)
        raw_data: 원본 분석 데이터 (y 포함)
        engine: AegisEngine 인스턴스
        synthesizer: FluidSynth 합성기
        sr: 샘플레이트
        hop_length: 홉 길이

    Returns:
        events: 검증된 이벤트 리스트
    """
    y_original = raw_data.get('y')
    if y_original is None:
        print("[TechniqueVerifier] ⚠️ 원본 오디오 없음. 검증 스킵.")
        return events

    verified_events = []

    for i, evt in enumerate(events):
        technique = evt.get('technique')

        # 테크닉이 있는 이벤트만 검증
        if technique in ['bend', 'hammer_on', 'pull_off']:
            print(f"[TechniqueVerifier] 🔍 검증 중: 노트 {evt['note']}, 테크닉 {technique}")

            # 시간 구간 계산
            start_sec = evt['start'] * hop_length / sr
            end_sec = evt['end'] * hop_length / sr
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)

            # 원본 오디오 구간 추출
            original_segment = y_original[start_sample:end_sample]

            if len(original_segment) < sr * 0.05:  # 50ms 미만은 스킵
                verified_events.append(evt)
                continue

            # 1. 테크닉 있는 버전 MIDI 생성
            with_technique_events = [evt]
            midi_with = _create_mini_midi(with_technique_events, sr, hop_length, engine)

            # 2. 테크닉 없는 버전 MIDI 생성 (일반 노트)
            evt_no_tech = evt.copy()
            evt_no_tech['technique'] = None
            evt_no_tech['slope'] = 0.0
            without_technique_events = [evt_no_tech]
            midi_without = _create_mini_midi(without_technique_events, sr, hop_length, engine)

            # 3. 두 버전 합성
            try:
                wav_with = synthesizer.midi_to_wav(midi_with, sample_rate=sr)
                wav_without = synthesizer.midi_to_wav(midi_without, sample_rate=sr)

                if wav_with is None or wav_without is None:
                    print(f"[TechniqueVerifier] ⚠️ 합성 실패. 원본 테크닉 유지.")
                    verified_events.append(evt)
                    continue

                # WAV 바이트 → numpy array 변환
                synth_with = _wav_bytes_to_audio(wav_with, sr)
                synth_without = _wav_bytes_to_audio(wav_without, sr)

                # 4. Mel spectrogram 유사도 비교
                similarity_with = _compute_similarity(original_segment, synth_with, sr)
                similarity_without = _compute_similarity(original_segment, synth_without, sr)

                print(f"[TechniqueVerifier] 유사도 - 테크닉 O: {similarity_with:.3f}, 테크닉 X: {similarity_without:.3f}")

                # 5. 유사도가 높은 버전 선택
                if similarity_with > similarity_without and similarity_with > 0.6:
                    # 테크닉 확정
                    verified_events.append(evt)
                    print(f"[TechniqueVerifier] ✅ 테크닉 '{technique}' 확정")
                else:
                    # 일반 노트로 변경
                    evt['technique'] = None
                    evt['slope'] = 0.0
                    verified_events.append(evt)
                    print(f"[TechniqueVerifier] ❌ 테크닉 '{technique}' 제거 → 일반 노트")

            except Exception as e:
                print(f"[TechniqueVerifier] ⚠️ 검증 실패: {e}")
                verified_events.append(evt)
        else:
            # 테크닉 없는 이벤트는 그대로 통과
            verified_events.append(evt)

    return verified_events


def _create_mini_midi(events, sr, hop_length, engine):
    """
    단일 이벤트 리스트로 미니 MIDI 파일 생성
    """
    import mido
    from mido import Message, MidiFile, MidiTrack
    import io

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Guitar Program
    track.append(Message('program_change', program=27, time=0))

    secs_per_frame = hop_length / sr
    ticks_per_sec = mido.second2tick(1.0, ticks_per_beat=480, tempo=500000)

    midi_events = []
    for evt in events:
        st = int(evt['start'] * secs_per_frame * ticks_per_sec)
        et = int(evt['end'] * secs_per_frame * ticks_per_sec)

        technique = evt.get('technique')
        velocity = evt['velocity']

        if technique == 'hammer_on':
            velocity = int(velocity * 0.6)
        elif technique == 'pull_off':
            velocity = int(velocity * 0.5)

        midi_events.append({'t': st, 'o': 'on', 'n': evt['note'], 'v': velocity})
        midi_events.append({'t': et, 'o': 'off', 'n': evt['note'], 'v': 0})

        # 벤딩 Pitch Bend
        if technique == 'bend':
            duration_ticks = et - st
            slope_value = evt.get('slope', 0.0)
            bend_semitones = min(2.0, abs(slope_value) * 10)
            bend_direction = 1 if slope_value > 0 else -1
            max_bend = int(bend_direction * (bend_semitones / 2.0) * 8191)

            num_bend_points = 15
            for i in range(num_bend_points):
                progress = i / num_bend_points
                curve = 1 - (1 - progress) ** 2
                bend_value = int(max_bend * curve)
                bend_tick = st + int(progress * duration_ticks)
                midi_events.append({'t': bend_tick, 'o': 'pitchwheel', 'pitch': bend_value})

            midi_events.append({'t': et, 'o': 'pitchwheel', 'pitch': 0})

    midi_events.sort(key=lambda x: x['t'])

    last_tick = 0
    for e in midi_events:
        if e['o'] == 'pitchwheel':
            track.append(Message('pitchwheel', pitch=e['pitch'], time=e['t'] - last_tick))
        elif e['o'] == 'on':
            track.append(Message('note_on', note=e['n'], velocity=e['v'], time=e['t'] - last_tick))
        elif e['o'] == 'off':
            track.append(Message('note_off', note=e['n'], velocity=0, time=e['t'] - last_tick))
        last_tick = e['t']

    # BytesIO로 반환
    midi_buffer = io.BytesIO()
    mid.save(file=midi_buffer)
    midi_buffer.seek(0)
    return midi_buffer.read()


def _wav_bytes_to_audio(wav_bytes, target_sr):
    """
    WAV 바이트 데이터를 numpy array로 변환
    """
    try:
        # BytesIO로 읽기
        audio, sr = sf.read(io.BytesIO(wav_bytes))

        # 모노로 변환
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # 샘플레이트 변환 (필요 시)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        return audio
    except Exception as e:
        print(f"[TechniqueVerifier] ⚠️ WAV 변환 실패: {e}")
        return None


def _compute_similarity(audio1, audio2, sr):
    """
    두 오디오 세그먼트의 Mel spectrogram 기반 코사인 유사도 계산
    """
    if audio1 is None or audio2 is None:
        return 0.0

    # 길이 맞추기
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]

    if len(audio1) < sr * 0.05:  # 50ms 미만
        return 0.0

    # Mel spectrogram 추출
    mel1 = librosa.feature.melspectrogram(y=audio1, sr=sr, n_mels=128, fmax=8000)
    mel2 = librosa.feature.melspectrogram(y=audio2, sr=sr, n_mels=128, fmax=8000)

    # dB 스케일
    mel1_db = librosa.power_to_db(mel1, ref=np.max)
    mel2_db = librosa.power_to_db(mel2, ref=np.max)

    # Flatten
    mel1_flat = mel1_db.flatten().reshape(1, -1)
    mel2_flat = mel2_db.flatten().reshape(1, -1)

    # 코사인 유사도
    similarity = cosine_similarity(mel1_flat, mel2_flat)[0][0]

    return similarity
