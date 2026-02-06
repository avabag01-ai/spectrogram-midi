"""
Aegis Engine - Per-Note Optimizer
노트별 개별 ADSR/신스 파라미터 최적화
각 노트를 원본 오디오 구간과 비교하여 최적의 음색 파라미터를 찾는다.

기존 ADSRSynthesizer가 곡 전체에 동일한 ADSR 파라미터를 적용하는 반면,
이 모듈은 개별 노트마다 원본 오디오 슬라이스와 비교하여
최적의 attack/decay/sustain/release + 파형 조합을 탐색한다.

워크플로우:
  1. 검출된 MIDI 이벤트(노트 목록) 수신
  2. 각 노트에 대해 원본 오디오에서 해당 구간 슬라이스
  3. 슬라이스의 엔벨로프(ADSR 특성) 분석
  4. 다양한 ADSR + 파형 조합으로 단일 노트 합성
  5. 합성 결과와 원본 슬라이스 비교 (유사도 측정)
  6. 해당 노트에 최적인 파라미터 선택
  7. 최종 출력: 각 노트별 최적화된 ADSR 파라미터 리스트
"""

import numpy as np
import librosa
import wave
import struct
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from aegis_engine_core.synthesizer import ADSRSynthesizer, get_adsr_synthesizer


# =============================================================================
# 1) 오디오 슬라이싱
# =============================================================================

def slice_audio_for_note(audio_data, sr, start_time, end_time, padding_ms=50):
    """
    단일 노트에 해당하는 오디오 구간을 추출한다.

    원본 오디오에서 노트의 시작~끝 시간에 해당하는 세그먼트를 잘라내며,
    전후로 약간의 패딩을 추가하여 문맥을 보존한다.

    Args:
        audio_data (numpy.ndarray): 원본 오디오 데이터 (1D float 배열)
        sr (int): 샘플 레이트 (Hz)
        start_time (float): 노트 시작 시간 (초)
        end_time (float): 노트 종료 시간 (초)
        padding_ms (int): 전후 패딩 (밀리초, 기본 50ms)

    Returns:
        numpy.ndarray: 슬라이스된 오디오 구간 (1D float 배열)
    """
    # 스테레오 → 모노 변환
    if audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=1)

    padding_samples = int(sr * padding_ms / 1000.0)

    start_sample = max(0, int(start_time * sr) - padding_samples)
    end_sample = min(len(audio_data), int(end_time * sr) + padding_samples)

    # 최소 길이 보장 (너무 짧으면 분석이 불가)
    if end_sample - start_sample < int(sr * 0.01):  # 최소 10ms
        end_sample = min(len(audio_data), start_sample + int(sr * 0.05))

    return audio_data[start_sample:end_sample].copy()


# =============================================================================
# 2) 노트 오디오 비교 (유사도 측정)
# =============================================================================

def compare_note_audio(original_slice, synthesized_slice, sr=44100):
    """
    원본 오디오 슬라이스와 합성된 슬라이스를 비교하여 유사도를 측정한다.

    세 가지 메트릭을 종합:
      - RMS 엔벨로프 상관관계 (시간적 에너지 변화 유사도)
      - 스펙트럼 센트로이드 유사도 (음색 밝기 유사도)
      - 제로 크로싱 레이트 유사도 (파형 복잡도 유사도)

    Args:
        original_slice (numpy.ndarray): 원본 오디오 슬라이스
        synthesized_slice (numpy.ndarray): 합성된 오디오 슬라이스
        sr (int): 샘플 레이트 (기본 44100Hz)

    Returns:
        float: 유사도 점수 (0.0 ~ 1.0, 1.0이 완전 일치)
    """
    # 길이 맞추기 (짧은 쪽에 제로패딩)
    max_len = max(len(original_slice), len(synthesized_slice))
    if max_len == 0:
        return 0.0

    orig = np.zeros(max_len)
    synth = np.zeros(max_len)
    orig[:len(original_slice)] = original_slice
    synth[:len(synthesized_slice)] = synthesized_slice

    # --- 메트릭 1: RMS 엔벨로프 상관관계 ---
    frame_length = max(512, int(sr * 0.01))  # 10ms 프레임
    hop_length = frame_length // 2

    rms_orig = librosa.feature.rms(
        y=orig, frame_length=frame_length, hop_length=hop_length
    )[0]
    rms_synth = librosa.feature.rms(
        y=synth, frame_length=frame_length, hop_length=hop_length
    )[0]

    rms_corr = 0.0
    if len(rms_orig) > 1 and np.std(rms_orig) > 1e-10 and np.std(rms_synth) > 1e-10:
        # numpy 상관계수 (-1 ~ 1) → 0 ~ 1 범위로 변환
        corr_matrix = np.corrcoef(rms_orig, rms_synth)
        rms_corr = float(np.clip((corr_matrix[0, 1] + 1.0) / 2.0, 0.0, 1.0))
    elif np.std(rms_orig) < 1e-10 and np.std(rms_synth) < 1e-10:
        # 둘 다 거의 무음이면 유사한 것으로 간주
        rms_corr = 1.0

    # --- 메트릭 2: 스펙트럼 센트로이드 유사도 ---
    centroid_sim = 0.0
    try:
        sc_orig = librosa.feature.spectral_centroid(y=orig, sr=sr)[0]
        sc_synth = librosa.feature.spectral_centroid(y=synth, sr=sr)[0]

        mean_orig = np.mean(sc_orig) if len(sc_orig) > 0 else 0.0
        mean_synth = np.mean(sc_synth) if len(sc_synth) > 0 else 0.0

        # 센트로이드 차이를 상대적 비율로 변환
        max_centroid = max(mean_orig, mean_synth, 1.0)
        centroid_diff = abs(mean_orig - mean_synth) / max_centroid
        centroid_sim = float(np.clip(1.0 - centroid_diff, 0.0, 1.0))
    except Exception:
        centroid_sim = 0.5  # 분석 실패 시 중립값

    # --- 메트릭 3: 제로 크로싱 레이트 유사도 ---
    zcr_sim = 0.0
    try:
        zcr_orig = librosa.feature.zero_crossing_rate(y=orig)[0]
        zcr_synth = librosa.feature.zero_crossing_rate(y=synth)[0]

        mean_zcr_orig = np.mean(zcr_orig) if len(zcr_orig) > 0 else 0.0
        mean_zcr_synth = np.mean(zcr_synth) if len(zcr_synth) > 0 else 0.0

        max_zcr = max(mean_zcr_orig, mean_zcr_synth, 1e-10)
        zcr_diff = abs(mean_zcr_orig - mean_zcr_synth) / max_zcr
        zcr_sim = float(np.clip(1.0 - zcr_diff, 0.0, 1.0))
    except Exception:
        zcr_sim = 0.5  # 분석 실패 시 중립값

    # --- 가중 평균으로 최종 유사도 산출 ---
    # RMS 엔벨로프가 가장 중요 (시간적 특성), 센트로이드 (음색), ZCR (파형)
    weights = {
        'rms_envelope': 0.50,
        'spectral_centroid': 0.30,
        'zero_crossing_rate': 0.20,
    }

    similarity = (
        weights['rms_envelope'] * rms_corr
        + weights['spectral_centroid'] * centroid_sim
        + weights['zero_crossing_rate'] * zcr_sim
    )

    return float(np.clip(similarity, 0.0, 1.0))


# =============================================================================
# 3) 단일 노트 최적화
# =============================================================================

def optimize_single_note(note_event, original_audio, sr=44100, quick_mode=True):
    """
    단일 노트에 대해 최적의 ADSR/파형 파라미터를 탐색한다.

    원본 오디오에서 해당 노트 구간을 슬라이스하고, ADSRSynthesizer의
    analyze_envelope()로 엔벨로프 특성을 추출한다.

    quick_mode=True: 분석된 엔벨로프를 그대로 사용 (빠름)
    quick_mode=False: 3 파형 x 3 어택 x 3 디케이 = 27개 조합을 시도하여
                      원본과 가장 유사한 조합 선택 (정밀하지만 느림)

    Args:
        note_event (dict): 노트 이벤트 정보
            - 'note' (int): MIDI 노트 번호
            - 'start' (int): 시작 프레임 인덱스
            - 'end' (int): 종료 프레임 인덱스
            - 'velocity' (int): MIDI 벨로시티 (0~127)
            - 'technique' (str): 연주 기법 (예: 'normal', 'bend', 'slide')
            - 'confidence' (float): 검출 신뢰도 (0.0~1.0)
        original_audio (numpy.ndarray): 원본 오디오 데이터 (1D float 배열)
        sr (int): 샘플 레이트 (기본 44100Hz)
        quick_mode (bool): True이면 빠른 분석 모드, False이면 그리드 서치

    Returns:
        dict: 최적화된 파라미터
            - 'attack_ms' (float): 최적 어택 시간 (밀리초)
            - 'decay_ms' (float): 최적 디케이 시간 (밀리초)
            - 'sustain_level' (float): 최적 서스테인 레벨 (0.0~1.0)
            - 'release_ms' (float): 최적 릴리스 시간 (밀리초)
            - 'waveform' (str): 최적 파형 종류
            - 'similarity_score' (float): 최종 유사도 점수 (0.0~1.0)
    """
    synth = get_adsr_synthesizer(sr=sr)
    hop_length = 512  # 기본 hop_length

    # 프레임 인덱스 → 시간(초) 변환
    start_time = note_event['start'] * hop_length / sr
    end_time = note_event['end'] * hop_length / sr
    duration = max(0.01, end_time - start_time)

    # 원본 오디오에서 해당 노트 구간 슬라이스
    original_slice = slice_audio_for_note(original_audio, sr, start_time, end_time)

    # 원본 슬라이스의 엔벨로프 분석
    analyzed_params = synth.analyze_envelope(original_slice, sr=sr)

    # MIDI 노트 → 주파수
    freq = 440.0 * (2.0 ** ((note_event['note'] - 69) / 12.0))
    velocity = note_event.get('velocity', 100)

    if quick_mode:
        # --- 빠른 모드: 분석된 엔벨로프 + 기본 파형(sawtooth) 그대로 사용 ---
        waveform = 'sawtooth'

        # 합성하여 유사도 측정 (확인용)
        full_duration = duration + analyzed_params['release_ms'] / 1000.0
        synthesized = synth.synthesize_note(
            freq=freq,
            duration=full_duration,
            velocity=velocity,
            attack_ms=analyzed_params['attack_ms'],
            decay_ms=analyzed_params['decay_ms'],
            sustain_level=analyzed_params['sustain_level'],
            release_ms=analyzed_params['release_ms'],
            waveform=waveform,
            harmonics=True,
        )

        # 원본 슬라이스 길이에 맞춰 자르기
        if len(synthesized) > len(original_slice):
            synthesized = synthesized[:len(original_slice)]

        similarity = compare_note_audio(original_slice, synthesized, sr=sr)

        return {
            'attack_ms': analyzed_params['attack_ms'],
            'decay_ms': analyzed_params['decay_ms'],
            'sustain_level': analyzed_params['sustain_level'],
            'release_ms': analyzed_params['release_ms'],
            'waveform': waveform,
            'similarity_score': round(similarity, 4),
        }

    else:
        # --- 정밀 모드: 그리드 서치 (27개 조합) ---
        waveforms = ['sawtooth', 'triangle', 'square']

        # 분석된 값을 기준으로 변이(variation) 생성
        base_attack = analyzed_params['attack_ms']
        base_decay = analyzed_params['decay_ms']
        base_sustain = analyzed_params['sustain_level']
        base_release = analyzed_params['release_ms']

        attack_variations = [
            max(1.0, base_attack * 0.5),
            base_attack,
            min(500.0, base_attack * 2.0),
        ]
        decay_variations = [
            max(1.0, base_decay * 0.5),
            base_decay,
            min(1000.0, base_decay * 2.0),
        ]

        best_params = None
        best_similarity = -1.0

        for wf in waveforms:
            for atk in attack_variations:
                for dcy in decay_variations:
                    # 합성
                    full_duration = duration + base_release / 1000.0
                    try:
                        synthesized = synth.synthesize_note(
                            freq=freq,
                            duration=full_duration,
                            velocity=velocity,
                            attack_ms=atk,
                            decay_ms=dcy,
                            sustain_level=base_sustain,
                            release_ms=base_release,
                            waveform=wf,
                            harmonics=True,
                        )
                    except Exception:
                        continue

                    # 길이 맞추기
                    if len(synthesized) > len(original_slice):
                        synthesized = synthesized[:len(original_slice)]

                    # 유사도 측정
                    sim = compare_note_audio(original_slice, synthesized, sr=sr)

                    if sim > best_similarity:
                        best_similarity = sim
                        best_params = {
                            'attack_ms': round(atk, 1),
                            'decay_ms': round(dcy, 1),
                            'sustain_level': round(base_sustain, 3),
                            'release_ms': round(base_release, 1),
                            'waveform': wf,
                            'similarity_score': round(sim, 4),
                        }

        # 그리드 서치 실패 시 분석값 폴백
        if best_params is None:
            best_params = {
                'attack_ms': analyzed_params['attack_ms'],
                'decay_ms': analyzed_params['decay_ms'],
                'sustain_level': analyzed_params['sustain_level'],
                'release_ms': analyzed_params['release_ms'],
                'waveform': 'sawtooth',
                'similarity_score': 0.0,
            }

        return best_params


# =============================================================================
# 4) 전체 노트 일괄 최적화 (메인 진입점)
# =============================================================================

def optimize_all_notes(events, original_audio, sr=44100, hop_length=512,
                       quick_mode=True, progress_callback=None):
    """
    모든 노트 이벤트에 대해 개별 ADSR 파라미터를 최적화한다.

    각 노트마다 optimize_single_note()를 호출하여 원본 오디오 구간과
    비교한 최적 파라미터를 구한다. 결과는 각 이벤트에 'adsr_params'
    필드로 추가된다.

    Args:
        events (list[dict]): 노트 이벤트 리스트
            각 이벤트: {'note', 'start', 'end', 'velocity', 'technique', 'confidence'}
        original_audio (numpy.ndarray): 원본 오디오 데이터 (1D float 배열)
        sr (int): 샘플 레이트 (기본 44100Hz)
        hop_length (int): 프레임→시간 변환용 hop 길이 (기본 512)
        quick_mode (bool): True이면 빠른 분석 모드 (기본값)
        progress_callback (callable, optional): 진행 상황 콜백 함수
            호출 시그니처: progress_callback(current, total, note_info)
            - current (int): 현재 처리 중인 노트 인덱스 (0-based)
            - total (int): 전체 노트 수
            - note_info (dict): 현재 노트 정보 (note, start, similarity 등)

    Returns:
        list[dict]: 'adsr_params' 필드가 추가된 이벤트 리스트
            각 이벤트에 추가되는 'adsr_params':
            {'attack_ms', 'decay_ms', 'sustain_level', 'release_ms',
             'waveform', 'similarity_score'}
    """
    if not events:
        return []

    # 스테레오 → 모노 변환 (한번만 수행)
    audio = original_audio
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    total = len(events)
    optimized_events = []

    for idx, event in enumerate(events):
        # 노트 이벤트를 깊은 복사하여 원본 보존
        opt_event = dict(event)

        try:
            # 개별 노트 최적화
            adsr_params = optimize_single_note(
                note_event=event,
                original_audio=audio,
                sr=sr,
                quick_mode=quick_mode,
            )
            opt_event['adsr_params'] = adsr_params

        except Exception as e:
            # 개별 노트 최적화 실패 시 기본값 사용
            opt_event['adsr_params'] = {
                'attack_ms': 10.0,
                'decay_ms': 50.0,
                'sustain_level': 0.7,
                'release_ms': 100.0,
                'waveform': 'sawtooth',
                'similarity_score': 0.0,
            }

        optimized_events.append(opt_event)

        # 진행 상황 콜백 호출
        if progress_callback is not None:
            note_info = {
                'note': event.get('note', 0),
                'start_frame': event.get('start', 0),
                'similarity': opt_event['adsr_params'].get('similarity_score', 0.0),
            }
            try:
                progress_callback(idx, total, note_info)
            except Exception:
                pass  # 콜백 에러는 무시

    return optimized_events


# =============================================================================
# 4-B) 멀티프로세싱 병렬 처리 버전
# =============================================================================

def _optimize_note_worker(args):
    """
    멀티프로세싱 워커 함수.

    ProcessPoolExecutor에서 호출되며, 단일 노트의 최적화를 수행한다.
    프로세스 간 pickle 직렬화를 위해 단순 dict/array만 주고받는다.

    Args:
        args: (idx, event_dict, audio_data, sr, quick_mode)

    Returns:
        (idx, adsr_params_dict)
    """
    idx, event, audio_data, sr, quick_mode = args
    try:
        adsr_params = optimize_single_note(
            note_event=event,
            original_audio=audio_data,
            sr=sr,
            quick_mode=quick_mode,
        )
        return (idx, adsr_params)
    except Exception as e:
        return (idx, {
            'attack_ms': 10.0,
            'decay_ms': 50.0,
            'sustain_level': 0.7,
            'release_ms': 100.0,
            'waveform': 'sawtooth',
            'similarity_score': 0.0,
        })


def optimize_all_notes_parallel(events, original_audio, sr=44100, hop_length=512,
                                 quick_mode=True, max_workers=None, progress_callback=None):
    """
    멀티프로세싱으로 모든 노트를 병렬 최적화한다.

    optimize_all_notes()와 동일한 기능이지만 CPU 코어를 활용하여
    노트별 최적화를 병렬로 수행한다. 노트 수가 많을수록 효과가 크다.

    Args:
        events (list[dict]): 노트 이벤트 리스트
        original_audio (numpy.ndarray): 원본 오디오 데이터
        sr (int): 샘플 레이트 (기본 44100Hz)
        hop_length (int): hop 길이 (기본 512)
        quick_mode (bool): 빠른 분석 모드 (기본 True)
        max_workers (int, optional): 최대 워커 수 (기본: CPU 코어 수)
        progress_callback (callable, optional): 진행 상황 콜백

    Returns:
        list[dict]: 'adsr_params' 필드가 추가된 이벤트 리스트
    """
    if not events:
        return []

    # 스테레오 → 모노
    audio = original_audio
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    total = len(events)

    # 노트가 적으면 순차 처리가 오히려 빠름 (프로세스 생성 오버헤드)
    if total < 10:
        print(f"[PerNoteOptimizer] 노트 {total}개 → 순차 처리 모드")
        return optimize_all_notes(events, audio, sr, hop_length, quick_mode, progress_callback)

    if max_workers is None:
        max_workers = min(mp.cpu_count(), total, 8)  # 최대 8워커

    print(f"[PerNoteOptimizer] 🚀 병렬 처리 모드: {total}개 노트, {max_workers}워커")

    # 워커 인자 준비
    worker_args = [
        (idx, dict(event), audio, sr, quick_mode)
        for idx, event in enumerate(events)
    ]

    # 결과 저장용
    results = [None] * total
    completed = 0

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_optimize_note_worker, args): args[0]
                for args in worker_args
            }

            for future in as_completed(futures):
                idx, adsr_params = future.result()
                results[idx] = adsr_params
                completed += 1

                if progress_callback:
                    try:
                        progress_callback(completed, total, {
                            'note': events[idx].get('note', 0),
                            'start_frame': events[idx].get('start', 0),
                            'similarity': adsr_params.get('similarity_score', 0.0),
                        })
                    except Exception:
                        pass

    except Exception as e:
        print(f"[PerNoteOptimizer] ⚠️ 병렬 처리 실패 ({e}), 순차 모드로 전환")
        return optimize_all_notes(events, audio, sr, hop_length, quick_mode, progress_callback)

    # 결과 조립
    optimized_events = []
    for idx, event in enumerate(events):
        opt_event = dict(event)
        opt_event['adsr_params'] = results[idx] or {
            'attack_ms': 10.0, 'decay_ms': 50.0,
            'sustain_level': 0.7, 'release_ms': 100.0,
            'waveform': 'sawtooth', 'similarity_score': 0.0,
        }
        optimized_events.append(opt_event)

    avg_sim = np.mean([e['adsr_params']['similarity_score'] for e in optimized_events])
    print(f"[PerNoteOptimizer] ✅ 완료! 평균 유사도: {avg_sim:.3f}")

    return optimized_events


# =============================================================================
# 5) 노트별 파라미터로 전체 오디오 합성
# =============================================================================

def synthesize_with_per_note_params(events, optimized_params, sr=44100):
    """
    각 노트에 개별 ADSR 파라미터를 적용하여 전체 오디오를 합성한다.

    ADSRSynthesizer.midi_to_wav()와 동일한 믹스다운 방식이지만,
    모든 노트에 동일한 파라미터를 사용하는 대신 각 노트마다
    최적화된 개별 파라미터를 적용한다.

    Args:
        events (list[dict]): 노트 이벤트 리스트
            각 이벤트: {'note', 'start', 'end', 'velocity', ...}
        optimized_params (list[dict]): 노트별 최적화 파라미터 리스트
            각 파라미터: {'attack_ms', 'decay_ms', 'sustain_level',
                         'release_ms', 'waveform', 'similarity_score'}
            events와 동일한 길이/순서여야 한다.
        sr (int): 샘플 레이트 (기본 44100Hz)

    Returns:
        bytes: WAV 파일 데이터 (16bit 모노)

    Raises:
        ValueError: events와 optimized_params의 길이가 다를 때
    """
    if len(events) != len(optimized_params):
        raise ValueError(
            f"events({len(events)})와 optimized_params({len(optimized_params)})의 "
            f"길이가 일치하지 않습니다."
        )

    if not events:
        # 빈 이벤트 → 1초 무음 WAV 반환
        silence = np.zeros(sr, dtype=np.int16)
        return _numpy_to_wav_bytes(silence, sr)

    synth = get_adsr_synthesizer(sr=sr)
    hop_length = 512

    # 전체 오디오 길이 계산
    max_end_time = 0.0
    for event in events:
        end_time = event['end'] * hop_length / sr
        max_end_time = max(max_end_time, end_time)

    # 릴리스 여유분 + 안전 마진
    max_release_ms = max(
        (p.get('release_ms', 100.0) for p in optimized_params),
        default=100.0,
    )
    total_duration = max_end_time + max_release_ms / 1000.0 + 0.5
    total_samples = int(sr * total_duration)
    mixed = np.zeros(total_samples, dtype=np.float64)

    # 각 노트를 개별 파라미터로 합성 후 믹스
    for event, params in zip(events, optimized_params):
        note_num = event.get('note', 60)
        velocity = event.get('velocity', 100)

        # 프레임 → 시간 변환
        start_time = event['start'] * hop_length / sr
        end_time = event['end'] * hop_length / sr
        duration = max(0.01, end_time - start_time)

        # MIDI 노트 → 주파수
        freq = 440.0 * (2.0 ** ((note_num - 69) / 12.0))

        # 개별 파라미터 추출
        attack_ms = params.get('attack_ms', 10.0)
        decay_ms = params.get('decay_ms', 50.0)
        sustain_level = params.get('sustain_level', 0.7)
        release_ms = params.get('release_ms', 100.0)
        waveform = params.get('waveform', 'sawtooth')

        # 릴리스 포함한 전체 노트 길이
        full_duration = duration + release_ms / 1000.0

        try:
            note_signal = synth.synthesize_note(
                freq=freq,
                duration=full_duration,
                velocity=velocity,
                attack_ms=attack_ms,
                decay_ms=decay_ms,
                sustain_level=sustain_level,
                release_ms=release_ms,
                waveform=waveform,
                harmonics=True,
            )
        except Exception:
            # 합성 실패 시 해당 노트 건너뜀
            continue

        # 시작 위치에 노트 배치
        start_sample = int(start_time * sr)
        end_sample = start_sample + len(note_signal)

        if end_sample > total_samples:
            note_signal = note_signal[:total_samples - start_sample]
            end_sample = total_samples

        if start_sample < total_samples and start_sample >= 0:
            mixed[start_sample:end_sample] += note_signal

    # 마스터 정규화 (클리핑 방지)
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.9  # 헤드룸 확보

    # float64 → int16 변환
    audio_int16 = np.clip(mixed * 32767, -32768, 32767).astype(np.int16)

    return _numpy_to_wav_bytes(audio_int16, sr)


def _numpy_to_wav_bytes(audio_int16, sr):
    """
    int16 numpy 배열을 WAV 바이트로 변환하는 내부 헬퍼.

    Args:
        audio_int16 (numpy.ndarray): int16 오디오 데이터
        sr (int): 샘플 레이트

    Returns:
        bytes: WAV 파일 데이터
    """
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)       # 모노
        wf.setsampwidth(2)       # 16bit
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    return wav_buffer.getvalue()


# =============================================================================
# 6) 최적화 결과 리포트 생성
# =============================================================================

def generate_optimization_report(optimized_events):
    """
    노트별 최적화 결과를 요약하는 통계 리포트를 생성한다.

    UI에서 최적화 결과를 시각적으로 표시할 때 사용하는 통계 데이터를 반환한다.

    Args:
        optimized_events (list[dict]): optimize_all_notes()의 반환값
            각 이벤트에 'adsr_params' 필드가 포함되어 있어야 한다.

    Returns:
        dict: 최적화 리포트
            - 'total_notes' (int): 전체 노트 수
            - 'avg_similarity' (float): 평균 유사도 점수
            - 'min_similarity' (float): 최저 유사도 점수
            - 'max_similarity' (float): 최고 유사도 점수
            - 'worst_notes' (list[dict]): 유사도가 가장 낮은 노트 상위 5개
                각 항목: {'note', 'start', 'similarity_score'}
            - 'waveform_distribution' (dict): 파형별 사용 빈도
                예: {'sawtooth': 45, 'triangle': 30, 'square': 10}
            - 'technique_distribution' (dict): 연주 기법별 노트 수
                예: {'normal': 60, 'bend': 5, 'slide': 3}
            - 'avg_attack_ms' (float): 평균 어택 시간
            - 'avg_decay_ms' (float): 평균 디케이 시간
            - 'avg_sustain_level' (float): 평균 서스테인 레벨
            - 'avg_release_ms' (float): 평균 릴리스 시간
    """
    if not optimized_events:
        return {
            'total_notes': 0,
            'avg_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0,
            'worst_notes': [],
            'waveform_distribution': {},
            'technique_distribution': {},
            'avg_attack_ms': 0.0,
            'avg_decay_ms': 0.0,
            'avg_sustain_level': 0.0,
            'avg_release_ms': 0.0,
        }

    # 유사도 점수 수집
    similarities = []
    waveform_counts = {}
    technique_counts = {}
    attack_values = []
    decay_values = []
    sustain_values = []
    release_values = []

    scored_notes = []  # (유사도, 노트정보) 튜플 리스트

    for event in optimized_events:
        params = event.get('adsr_params', {})
        sim = params.get('similarity_score', 0.0)
        similarities.append(sim)

        # 파형 분포
        wf = params.get('waveform', 'unknown')
        waveform_counts[wf] = waveform_counts.get(wf, 0) + 1

        # 연주 기법 분포
        tech = event.get('technique', 'unknown')
        technique_counts[tech] = technique_counts.get(tech, 0) + 1

        # ADSR 통계
        attack_values.append(params.get('attack_ms', 10.0))
        decay_values.append(params.get('decay_ms', 50.0))
        sustain_values.append(params.get('sustain_level', 0.7))
        release_values.append(params.get('release_ms', 100.0))

        # worst 노트 후보
        scored_notes.append({
            'note': event.get('note', 0),
            'start': event.get('start', 0),
            'similarity_score': sim,
        })

    # 유사도 낮은 순으로 정렬하여 상위 5개 추출
    scored_notes.sort(key=lambda x: x['similarity_score'])
    worst_notes = scored_notes[:5]

    return {
        'total_notes': len(optimized_events),
        'avg_similarity': round(float(np.mean(similarities)), 4),
        'min_similarity': round(float(np.min(similarities)), 4),
        'max_similarity': round(float(np.max(similarities)), 4),
        'worst_notes': worst_notes,
        'waveform_distribution': waveform_counts,
        'technique_distribution': technique_counts,
        'avg_attack_ms': round(float(np.mean(attack_values)), 1),
        'avg_decay_ms': round(float(np.mean(decay_values)), 1),
        'avg_sustain_level': round(float(np.mean(sustain_values)), 3),
        'avg_release_ms': round(float(np.mean(release_values)), 1),
    }
