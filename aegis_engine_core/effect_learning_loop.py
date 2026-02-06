"""
Aegis Engine - Effect Learning Loop (역학습 루프)

알려진 MIDI(정답)로부터 역방향 학습을 수행하는 모듈.
다양한 오디오 이펙트를 적용한 후 엔진의 MIDI 변환 정확도를 측정하고,
반복적으로 엔진 파라미터를 조정하여 최적의 변환 품질을 달성한다.

학습 흐름:
    1. MIDI → WAV (FluidSynth 합성)
    2. WAV → 이펙트 적용 (distortion, reverb, delay, chorus)
    3. 이펙트 적용 WAV → 엔진 audio_to_midi() → extract_events() → 새 MIDI
    4. 원본 MIDI 노트 vs 새 MIDI 노트 비교 (피치, 타이밍, 노트 수)
    5. 파라미터 조정: confidence_threshold, min_note_duration_ms, sustain_ms
    6. 수렴 또는 최대 반복까지 반복

이펙트 체인은 순수 numpy/scipy로 구현하여 외부 이펙트 라이브러리에 의존하지 않는다.
"""

import numpy as np
import tempfile
import os
import io
import wave
import struct
import mido

from aegis_engine_core.synthesizer import synthesize_midi


# ============================================================================
# 이펙트 프리셋 설정
# ============================================================================

EFFECT_PRESETS = {
    'clean': [],
    'light_overdrive': [('distortion', {'drive': 0.3})],
    'heavy_distortion': [('distortion', {'drive': 0.8})],
    'ambient': [
        ('reverb', {'room_size': 0.7}),
        ('delay', {'delay_ms': 400, 'feedback': 0.3})
    ],
    'chorus_clean': [('chorus', {'depth': 0.003, 'rate': 1.5})],
    'full_fx': [
        ('distortion', {'drive': 0.4}),
        ('chorus', {'depth': 0.002}),
        ('reverb', {'room_size': 0.5}),
        ('delay', {'delay_ms': 300, 'feedback': 0.2})
    ],
}


# ============================================================================
# 오디오 이펙트 함수 (순수 numpy/scipy 구현)
# ============================================================================

def apply_distortion(audio, drive=0.5):
    """
    tanh 클리핑 기반 디스토션 이펙트.

    드라이브 값이 높을수록 더 강한 클리핑(왜곡)이 적용된다.
    tanh 함수를 사용하여 부드러운 클리핑을 구현한다.

    Args:
        audio (np.ndarray): 입력 오디오 신호 (-1.0 ~ 1.0 범위 float)
        drive (float): 드라이브 강도 (0.0 ~ 1.0). 기본값 0.5.
                        0에 가까우면 클린, 1에 가까우면 강한 왜곡.

    Returns:
        np.ndarray: 디스토션이 적용된 오디오 신호
    """
    # drive를 게인으로 변환 (1.0 ~ 20.0 범위)
    gain = 1.0 + drive * 19.0

    # tanh 소프트 클리핑 적용
    distorted = np.tanh(audio * gain)

    # 출력 레벨 보정 (드라이브가 높을수록 음량이 커지는 것 방지)
    distorted = distorted * (1.0 / max(np.max(np.abs(distorted)), 1e-6))
    distorted = np.clip(distorted, -1.0, 1.0)

    return distorted


def apply_reverb(audio, room_size=0.5, sr=44100):
    """
    간단한 컨볼루션 리버브 이펙트.

    지수 감쇠 임펄스 응답(IR)을 생성하고 입력 신호와 컨볼루션하여
    공간감 있는 리버브 효과를 만든다.

    Args:
        audio (np.ndarray): 입력 오디오 신호
        room_size (float): 방 크기 (0.0 ~ 1.0). 기본값 0.5.
                           클수록 더 긴 잔향이 생긴다.
        sr (int): 샘플링 레이트. 기본값 44100Hz.

    Returns:
        np.ndarray: 리버브가 적용된 오디오 신호
    """
    # 리버브 지속 시간 (방 크기에 비례, 최대 3초)
    reverb_duration = room_size * 3.0
    ir_length = int(sr * reverb_duration)

    if ir_length <= 0:
        return audio.copy()

    # 지수 감쇠 임펄스 응답 생성
    t = np.arange(ir_length, dtype=np.float64)
    decay_rate = 5.0 / max(reverb_duration, 0.01)  # 감쇠 속도
    impulse_response = np.exp(-decay_rate * t / sr)

    # 약간의 랜덤 디퓨전 추가 (자연스러운 잔향)
    rng = np.random.RandomState(42)  # 재현 가능한 랜덤
    impulse_response *= rng.uniform(0.8, 1.0, size=ir_length)

    # 임펄스 응답 정규화
    impulse_response /= max(np.sum(np.abs(impulse_response)), 1e-6)

    # 컨볼루션을 통한 리버브 적용
    # scipy가 없을 수 있으므로 numpy.convolve 사용
    wet = np.convolve(audio, impulse_response, mode='full')[:len(audio)]

    # 드라이/웻 믹스 (room_size에 비례하여 웻 비율 증가)
    wet_ratio = room_size * 0.6  # 최대 60% 웻
    dry_ratio = 1.0 - wet_ratio * 0.5  # 드라이 약간 줄임

    mixed = dry_ratio * audio + wet_ratio * wet

    # 클리핑 방지 정규화
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed /= max_val

    return mixed


def apply_delay(audio, delay_ms=300, feedback=0.3, sr=44100):
    """
    에코/딜레이 이펙트.

    지정된 시간 간격으로 원본 신호를 반복하여 에코 효과를 만든다.
    피드백 값에 따라 반복 횟수와 감쇠가 결정된다.

    Args:
        audio (np.ndarray): 입력 오디오 신호
        delay_ms (float): 딜레이 시간 (밀리초). 기본값 300ms.
        feedback (float): 피드백 양 (0.0 ~ 1.0). 기본값 0.3.
                          클수록 에코가 오래 지속된다.
        sr (int): 샘플링 레이트. 기본값 44100Hz.

    Returns:
        np.ndarray: 딜레이가 적용된 오디오 신호
    """
    delay_samples = int((delay_ms / 1000.0) * sr)

    if delay_samples <= 0 or feedback <= 0:
        return audio.copy()

    # 출력 버퍼 (원본보다 약간 길게)
    output = audio.copy().astype(np.float64)

    # 최대 반복 횟수 (피드백이 충분히 감쇠될 때까지)
    max_echoes = int(np.log(0.01) / np.log(max(feedback, 0.01)))
    max_echoes = min(max_echoes, 20)  # 안전 상한

    for i in range(1, max_echoes + 1):
        offset = delay_samples * i
        gain = feedback ** i

        if offset >= len(output) or gain < 0.01:
            break

        # 에코 신호 겹치기
        echo_length = min(len(audio), len(output) - offset)
        output[offset:offset + echo_length] += audio[:echo_length] * gain

    # 클리핑 방지 정규화
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output /= max_val

    return output


def apply_chorus(audio, depth=0.003, rate=1.5, sr=44100):
    """
    LFO 기반 피치 변조 코러스 이펙트.

    저주파 발진기(LFO)로 딜레이 시간을 변조하여 코러스 효과를 만든다.
    미세한 피치 변화가 더해져 소리가 풍성해진다.

    Args:
        audio (np.ndarray): 입력 오디오 신호
        depth (float): 변조 깊이 (초 단위). 기본값 0.003 (3ms).
                       클수록 더 뚜렷한 코러스 효과.
        rate (float): LFO 주파수 (Hz). 기본값 1.5Hz.
                      변조 속도를 결정한다.
        sr (int): 샘플링 레이트. 기본값 44100Hz.

    Returns:
        np.ndarray: 코러스가 적용된 오디오 신호
    """
    n_samples = len(audio)
    t = np.arange(n_samples, dtype=np.float64)

    # LFO로 변조된 딜레이 시간 (샘플 단위)
    base_delay = int(0.007 * sr)  # 기본 딜레이 7ms
    depth_samples = depth * sr
    lfo = np.sin(2.0 * np.pi * rate * t / sr)
    delay_modulated = base_delay + depth_samples * lfo

    # 변조된 딜레이를 사용해 샘플 인덱스 계산
    indices = t - delay_modulated
    indices = np.clip(indices, 0, n_samples - 1)

    # 선형 보간으로 부드러운 샘플 읽기
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, n_samples - 1)
    frac = indices - idx_floor

    chorus_signal = audio[idx_floor] * (1.0 - frac) + audio[idx_ceil] * frac

    # 드라이/웻 믹스 (50:50)
    output = 0.7 * audio + 0.3 * chorus_signal

    # 클리핑 방지
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output /= max_val

    return output


def apply_effect_chain(audio, effects_config, sr=44100):
    """
    여러 이펙트를 순서대로 체이닝하여 적용.

    effects_config 리스트에 정의된 순서대로 이펙트를 적용한다.
    각 이펙트는 (이펙트명, 파라미터딕셔너리) 튜플로 정의된다.

    Args:
        audio (np.ndarray): 입력 오디오 신호
        effects_config (list): 이펙트 설정 리스트.
            예: [('distortion', {'drive': 0.5}), ('reverb', {'room_size': 0.7})]
        sr (int): 샘플링 레이트. 기본값 44100Hz.

    Returns:
        np.ndarray: 이펙트 체인이 적용된 오디오 신호
    """
    # 이펙트 함수 매핑
    effect_functions = {
        'distortion': apply_distortion,
        'reverb': apply_reverb,
        'delay': apply_delay,
        'chorus': apply_chorus,
    }

    processed = audio.copy().astype(np.float64)

    for effect_name, params in effects_config:
        if effect_name not in effect_functions:
            print(f"[EffectLearningLoop] 알 수 없는 이펙트: {effect_name}, 건너뜁니다.")
            continue

        effect_fn = effect_functions[effect_name]

        # sr 파라미터가 필요한 이펙트에 자동으로 추가
        if effect_name in ('reverb', 'delay', 'chorus'):
            params_with_sr = {**params, 'sr': sr}
        else:
            params_with_sr = params

        processed = effect_fn(processed, **params_with_sr)

    return processed


# ============================================================================
# WAV 유틸리티 함수
# ============================================================================

def _wav_bytes_to_float(wav_data):
    """
    WAV 바이트 데이터를 float 오디오 배열로 변환.

    Args:
        wav_data (bytes): WAV 파일 바이트 데이터

    Returns:
        tuple: (audio_float_array, sample_rate, n_channels)
    """
    with io.BytesIO(wav_data) as buf:
        with wave.open(buf, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw_frames = wf.readframes(n_frames)

    # 바이트를 정수 배열로 변환
    if sample_width == 2:
        fmt = f'<{n_frames * n_channels}h'
        samples = np.array(struct.unpack(fmt, raw_frames), dtype=np.float64)
        samples /= 32768.0  # 16bit 정규화
    elif sample_width == 4:
        fmt = f'<{n_frames * n_channels}i'
        samples = np.array(struct.unpack(fmt, raw_frames), dtype=np.float64)
        samples /= 2147483648.0  # 32bit 정규화
    elif sample_width == 1:
        samples = np.array(list(raw_frames), dtype=np.float64)
        samples = (samples - 128.0) / 128.0  # 8bit 정규화
    else:
        raise ValueError(f"지원하지 않는 샘플 폭: {sample_width} bytes")

    # 스테레오인 경우 모노로 변환
    if n_channels == 2:
        samples = (samples[0::2] + samples[1::2]) / 2.0

    return samples, sr, n_channels


def _float_to_wav_bytes(audio, sr=44100):
    """
    float 오디오 배열을 WAV 바이트 데이터로 변환.

    Args:
        audio (np.ndarray): float 오디오 신호 (-1.0 ~ 1.0)
        sr (int): 샘플링 레이트

    Returns:
        bytes: WAV 파일 바이트 데이터
    """
    # 클리핑 방지
    audio = np.clip(audio, -1.0, 1.0)

    # float → 16bit int 변환
    int_samples = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)      # 모노
        wf.setsampwidth(2)      # 16bit
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())

    return buf.getvalue()


# ============================================================================
# MIDI 노트 추출 및 비교 (reverse_analyzer.py 패턴 재사용)
# ============================================================================

def _extract_notes_from_midi(midi_data):
    """
    MIDI 바이트 데이터에서 노트 정보 추출.

    note_on / note_off 메시지를 파싱하여 각 노트의 피치, 시작/종료 시간,
    벨로시티 정보를 추출한다.

    Args:
        midi_data (bytes 또는 BytesIO): MIDI 파일 데이터

    Returns:
        list: [{'pitch': int, 'start_time': float, 'end_time': float, 'velocity': int}, ...]
              빈 리스트: 추출 실패 시
    """
    try:
        if isinstance(midi_data, bytes):
            midi_data = io.BytesIO(midi_data)

        mid = mido.MidiFile(file=midi_data)

        notes = []
        ticks_per_beat = mid.ticks_per_beat
        tempo = 500000  # 기본 템포 (120 BPM)

        for track in mid.tracks:
            current_time = 0
            active_notes = {}  # {pitch: (start_time, velocity)}

            for msg in track:
                current_time += msg.time

                if msg.type == 'set_tempo':
                    tempo = msg.tempo

                elif msg.type == 'note_on' and msg.velocity > 0:
                    time_sec = mido.tick2second(current_time, ticks_per_beat, tempo)
                    active_notes[msg.note] = (time_sec, msg.velocity)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time, velocity = active_notes.pop(msg.note)
                        end_time = mido.tick2second(current_time, ticks_per_beat, tempo)

                        notes.append({
                            'pitch': msg.note,
                            'start_time': start_time,
                            'end_time': end_time,
                            'velocity': velocity
                        })

        return notes

    except Exception as e:
        print(f"[EffectLearningLoop] MIDI 노트 추출 실패: {e}")
        return []


def _compare_note_lists(original_notes, reversed_notes, time_tolerance=0.1, pitch_tolerance=1):
    """
    두 노트 리스트의 유사도 계산.

    원본 노트 각각에 대해 역변환된 노트 중 가장 가까운 매치를 찾고,
    피치 정확도, 타이밍 정확도, 전체 노트 일치율을 계산한다.

    Args:
        original_notes (list): 원본 MIDI 노트 리스트
        reversed_notes (list): 역변환된 MIDI 노트 리스트
        time_tolerance (float): 시간 허용 오차 (초). 기본값 0.1초.
        pitch_tolerance (int): 피치 허용 오차 (반음). 기본값 1.

    Returns:
        dict: {
            'note_accuracy': float,   # 노트 일치율 (0.0~1.0)
            'pitch_accuracy': float,  # 피치 정확도 (0.0~1.0)
            'timing_accuracy': float  # 타이밍 정확도 (0.0~1.0)
        }
    """
    if not original_notes or not reversed_notes:
        return {
            'note_accuracy': 0.0,
            'pitch_accuracy': 0.0,
            'timing_accuracy': 0.0
        }

    matched_count = 0
    pitch_errors = []
    timing_errors = []

    # 각 원본 노트에 대해 가장 가까운 역변환 노트 찾기
    for orig in original_notes:
        best_match = None
        best_distance = float('inf')

        for rev in reversed_notes:
            pitch_diff = abs(orig['pitch'] - rev['pitch'])
            time_diff = abs(orig['start_time'] - rev['start_time'])

            # 종합 거리 (피치는 12반음=1옥타브 단위로 정규화)
            distance = (pitch_diff / 12.0) + time_diff

            if distance < best_distance:
                best_distance = distance
                best_match = rev

        if best_match:
            pitch_diff = abs(orig['pitch'] - best_match['pitch'])
            time_diff = abs(orig['start_time'] - best_match['start_time'])

            if pitch_diff <= pitch_tolerance and time_diff <= time_tolerance:
                matched_count += 1

            pitch_errors.append(pitch_diff)
            timing_errors.append(time_diff)

    # 노트 일치율
    note_accuracy = matched_count / len(original_notes)

    # 피치 정확도: 평균 오차의 역수 (1옥타브 기준)
    avg_pitch_error = np.mean(pitch_errors) if pitch_errors else 12.0
    pitch_accuracy = max(0.0, 1.0 - (avg_pitch_error / 12.0))

    # 타이밍 정확도: 평균 오차의 역수 (0.5초 기준)
    avg_timing_error = np.mean(timing_errors) if timing_errors else 1.0
    timing_accuracy = max(0.0, 1.0 - (avg_timing_error / 0.5))

    return {
        'note_accuracy': note_accuracy,
        'pitch_accuracy': pitch_accuracy,
        'timing_accuracy': timing_accuracy
    }


# ============================================================================
# 메인 학습 루프
# ============================================================================

def learning_loop(
    midi_data,
    engine,
    effects_config,
    max_iterations=5,
    target_accuracy=0.95,
    progress_callback=None
):
    """
    역학습 루프 메인 함수.

    알려진 MIDI(정답)로 WAV를 합성하고, 이펙트를 적용한 뒤,
    엔진으로 다시 MIDI를 추출하여 정답과 비교한다.
    반복적으로 엔진 파라미터를 조정하여 최적의 변환 정확도를 달성한다.

    Args:
        midi_data (bytes): 원본 MIDI 바이트 데이터 (ground truth)
        engine: AegisEngine 인스턴스.
                engine.audio_to_midi(wav_path, None, turbo_mode=False) 메서드와
                engine.extract_events(raw_data, midi_buffer, ...) 메서드를 가져야 한다.
        effects_config (list): 적용할 이펙트 설정 리스트.
            EFFECT_PRESETS의 값 또는 커스텀 리스트.
            예: [('distortion', {'drive': 0.5}), ('reverb', {'room_size': 0.6})]
        max_iterations (int): 최대 반복 횟수. 기본값 5.
        target_accuracy (float): 목표 정확도 (0.0 ~ 1.0). 기본값 0.95.
                                 이 값에 도달하면 조기 종료한다.
        progress_callback (callable, optional): 각 반복마다 호출되는 콜백 함수.
            서명: progress_callback(iteration, max_iterations, accuracy_dict)

    Returns:
        dict: {
            'best_params': {
                'confidence_threshold': float,
                'min_note_duration_ms': int,
                'sustain_ms': int
            },
            'best_accuracy': {
                'note_accuracy': float,
                'pitch_accuracy': float,
                'timing_accuracy': float,
                'overall': float  # 가중 평균
            },
            'history': [  # 각 반복의 기록
                {'iteration': int, 'params': dict, 'accuracy': dict}
            ],
            'effect_profile': str  # 사용된 이펙트 설정 이름
        }
        None: 학습 실패 시
    """
    print("[EffectLearningLoop] === 역학습 루프 시작 ===")

    # ---- 이펙트 프로파일 이름 결정 ----
    effect_profile = _identify_effect_profile(effects_config)
    print(f"[EffectLearningLoop] 이펙트 프로파일: {effect_profile}")

    # ---- 1단계: 원본 MIDI 노트 추출 ----
    print("[EffectLearningLoop] 원본 MIDI 노트 추출 중...")
    original_notes = _extract_notes_from_midi(midi_data)

    if not original_notes:
        print("[EffectLearningLoop] 원본 MIDI에 노트가 없습니다. 중단.")
        return None

    print(f"[EffectLearningLoop] 원본 노트 수: {len(original_notes)}")

    # ---- 2단계: MIDI → WAV 합성 ----
    print("[EffectLearningLoop] MIDI → WAV 합성 중...")
    wav_data = synthesize_midi(midi_data, sample_rate=44100)

    if not wav_data:
        print("[EffectLearningLoop] MIDI 합성 실패. 중단.")
        return None

    # ---- 3단계: WAV → float 배열 변환 ----
    try:
        audio_float, sr, _ = _wav_bytes_to_float(wav_data)
    except Exception as e:
        print(f"[EffectLearningLoop] WAV 변환 실패: {e}")
        return None

    # ---- 4단계: 이펙트 적용 ----
    print(f"[EffectLearningLoop] 이펙트 체인 적용 중 ({len(effects_config)}개 이펙트)...")
    effected_audio = apply_effect_chain(audio_float, effects_config, sr=sr)

    # ---- 5단계: 이펙트 적용된 WAV를 임시 파일로 저장 ----
    effected_wav_data = _float_to_wav_bytes(effected_audio, sr=sr)

    # ---- 초기 파라미터 설정 ----
    params = {
        'confidence_threshold': 0.3,
        'min_note_duration_ms': 50,
        'sustain_ms': 200,
    }

    best_params = params.copy()
    best_accuracy = {
        'note_accuracy': 0.0,
        'pitch_accuracy': 0.0,
        'timing_accuracy': 0.0,
        'overall': 0.0,
    }
    history = []

    # ---- 6단계: 반복 학습 루프 ----
    for iteration in range(1, max_iterations + 1):
        print(f"\n[EffectLearningLoop] --- 반복 {iteration}/{max_iterations} ---")
        print(f"  파라미터: confidence={params['confidence_threshold']:.3f}, "
              f"min_duration={params['min_note_duration_ms']}ms, "
              f"sustain={params['sustain_ms']}ms")

        # 이펙트 적용된 WAV → 임시 파일
        tmp_wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_wav.write(effected_wav_data)
                tmp_wav_path = tmp_wav.name

            # 엔진으로 MIDI 역변환
            raw_data = engine.audio_to_midi(tmp_wav_path, None, turbo_mode=False)

            if not raw_data:
                print(f"  [반복 {iteration}] audio_to_midi 실패, 건너뜁니다.")
                history.append({
                    'iteration': iteration,
                    'params': params.copy(),
                    'accuracy': {
                        'note_accuracy': 0.0,
                        'pitch_accuracy': 0.0,
                        'timing_accuracy': 0.0,
                        'overall': 0.0,
                    }
                })
                continue

            # MIDI 이벤트 추출 (현재 파라미터 적용)
            reversed_midi_buffer = io.BytesIO()
            reversed_events = engine.extract_events(
                raw_data,
                reversed_midi_buffer,
                confidence_threshold=params['confidence_threshold'],
                min_note_duration_ms=params['min_note_duration_ms'],
                sustain_ms=params['sustain_ms'],
                midi_program=27
            )

            reversed_midi_buffer.seek(0)
            reversed_midi_data = reversed_midi_buffer.read()

            # 역변환 MIDI 노트 추출
            reversed_notes = _extract_notes_from_midi(reversed_midi_data)
            print(f"  역변환 노트 수: {len(reversed_notes)}")

            # 정확도 비교
            comparison = _compare_note_lists(original_notes, reversed_notes)

            # 가중 평균 계산 (노트 일치율 50%, 피치 30%, 타이밍 20%)
            overall = (
                comparison['note_accuracy'] * 0.5 +
                comparison['pitch_accuracy'] * 0.3 +
                comparison['timing_accuracy'] * 0.2
            )

            accuracy = {
                'note_accuracy': comparison['note_accuracy'],
                'pitch_accuracy': comparison['pitch_accuracy'],
                'timing_accuracy': comparison['timing_accuracy'],
                'overall': overall,
            }

            print(f"  노트 일치율: {accuracy['note_accuracy']:.1%}")
            print(f"  피치 정확도: {accuracy['pitch_accuracy']:.1%}")
            print(f"  타이밍 정확도: {accuracy['timing_accuracy']:.1%}")
            print(f"  종합 정확도: {accuracy['overall']:.1%}")

            # 기록 저장
            history.append({
                'iteration': iteration,
                'params': params.copy(),
                'accuracy': accuracy.copy(),
            })

            # 최고 정확도 갱신
            if overall > best_accuracy['overall']:
                best_accuracy = accuracy.copy()
                best_params = params.copy()
                print(f"  [최고 기록 갱신!] overall={overall:.1%}")

            # 진행 콜백 호출
            if progress_callback:
                try:
                    progress_callback(iteration, max_iterations, accuracy)
                except Exception:
                    pass  # 콜백 오류는 무시

            # 목표 달성 시 조기 종료
            if overall >= target_accuracy:
                print(f"\n[EffectLearningLoop] 목표 정확도 달성! ({overall:.1%} >= {target_accuracy:.1%})")
                break

            # ---- 파라미터 조정 (다음 반복을 위해) ----
            params = _adjust_parameters(
                params, accuracy, original_notes, reversed_notes
            )

        except Exception as e:
            print(f"  [반복 {iteration}] 오류 발생: {e}")
            history.append({
                'iteration': iteration,
                'params': params.copy(),
                'accuracy': {
                    'note_accuracy': 0.0,
                    'pitch_accuracy': 0.0,
                    'timing_accuracy': 0.0,
                    'overall': 0.0,
                }
            })

        finally:
            # 임시 파일 정리
            if tmp_wav_path:
                try:
                    os.unlink(tmp_wav_path)
                except OSError:
                    pass

    # ---- 결과 반환 ----
    print(f"\n[EffectLearningLoop] === 학습 완료 ===")
    print(f"  최적 파라미터: {best_params}")
    print(f"  최고 종합 정확도: {best_accuracy['overall']:.1%}")
    print(f"  총 반복 횟수: {len(history)}")

    return {
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'history': history,
        'effect_profile': effect_profile,
    }


# ============================================================================
# 내부 헬퍼 함수
# ============================================================================

def _identify_effect_profile(effects_config):
    """
    이펙트 설정 리스트를 프리셋 이름과 매칭하여 프로파일 이름을 반환.

    Args:
        effects_config (list): 이펙트 설정 리스트

    Returns:
        str: 매칭되는 프리셋 이름 또는 'custom'
    """
    for name, preset in EFFECT_PRESETS.items():
        if effects_config == preset:
            return name
    return 'custom'


def _adjust_parameters(params, accuracy, original_notes, reversed_notes):
    """
    현재 정확도를 기반으로 엔진 파라미터를 휴리스틱하게 조정.

    조정 전략:
    - 노트 수가 적으면 → confidence_threshold를 낮춰 더 많은 노트 검출
    - 노트 수가 많으면 → confidence_threshold를 높여 노이즈 노트 제거
    - 타이밍 정확도가 낮으면 → min_note_duration_ms 조정
    - 피치 정확도가 낮으면 → sustain_ms 조정

    Args:
        params (dict): 현재 파라미터
        accuracy (dict): 현재 정확도
        original_notes (list): 원본 노트 리스트
        reversed_notes (list): 역변환 노트 리스트

    Returns:
        dict: 조정된 파라미터
    """
    new_params = params.copy()

    orig_count = len(original_notes)
    rev_count = len(reversed_notes)

    # ---- confidence_threshold 조정 ----
    # 노트 수 비율에 따라 조정
    if orig_count > 0 and rev_count > 0:
        count_ratio = rev_count / orig_count

        if count_ratio < 0.7:
            # 노트가 너무 적게 검출됨 → 임계값 낮춤
            new_params['confidence_threshold'] = max(
                0.1,
                params['confidence_threshold'] - 0.05
            )
        elif count_ratio > 1.5:
            # 노트가 너무 많이 검출됨 → 임계값 높임
            new_params['confidence_threshold'] = min(
                0.8,
                params['confidence_threshold'] + 0.05
            )
    elif rev_count == 0:
        # 노트가 전혀 없음 → 크게 낮춤
        new_params['confidence_threshold'] = max(
            0.1,
            params['confidence_threshold'] - 0.1
        )

    # ---- min_note_duration_ms 조정 ----
    if accuracy['timing_accuracy'] < 0.5:
        # 타이밍 정확도가 낮으면 짧은 노트를 더 많이 허용
        new_params['min_note_duration_ms'] = max(
            20,
            params['min_note_duration_ms'] - 10
        )
    elif accuracy['note_accuracy'] > 0.8 and accuracy['timing_accuracy'] < 0.7:
        # 노트는 잘 맞지만 타이밍이 부정확 → 최소 지속 시간 미세 조정
        new_params['min_note_duration_ms'] = max(
            20,
            params['min_note_duration_ms'] - 5
        )

    # ---- sustain_ms 조정 ----
    if accuracy['pitch_accuracy'] < 0.5:
        # 피치 정확도가 낮으면 서스테인 줄임 (노트 분리 개선)
        new_params['sustain_ms'] = max(
            50,
            params['sustain_ms'] - 30
        )
    elif accuracy['note_accuracy'] < 0.5:
        # 전반적으로 정확도가 낮으면 서스테인 늘림
        new_params['sustain_ms'] = min(
            500,
            params['sustain_ms'] + 30
        )

    # 파라미터가 변경되었는지 확인
    if new_params == params:
        # 변경이 없으면 약간의 랜덤 탐색 (지역 최솟값 탈출)
        rng = np.random.RandomState()
        new_params['confidence_threshold'] = np.clip(
            params['confidence_threshold'] + rng.uniform(-0.03, 0.03),
            0.1, 0.8
        )
        new_params['min_note_duration_ms'] = int(np.clip(
            params['min_note_duration_ms'] + rng.randint(-5, 6),
            20, 200
        ))
        new_params['sustain_ms'] = int(np.clip(
            params['sustain_ms'] + rng.randint(-20, 21),
            50, 500
        ))

    return new_params
