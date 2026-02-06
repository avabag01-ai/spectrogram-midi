"""
Aegis Engine - Auto Parameter Matcher
원본 음원과 MIDI 합성 결과를 비교해서 최적 파라미터를 자동 탐색
"""

import numpy as np
import librosa
import io
import tempfile
from aegis_engine_core.synthesizer import synthesize_midi


def _calculate_similarity(original_audio_path, synthesized_wav_data, sample_rate=44100):
    """
    원본 음원과 합성된 WAV의 유사도를 계산

    Args:
        original_audio_path: 원본 음원 파일 경로
        synthesized_wav_data: 합성된 WAV 바이트 데이터
        sample_rate: 샘플링 레이트

    Returns:
        float: 유사도 스코어 (0.0~1.0, 높을수록 유사)
    """
    try:
        # 원본 음원 로드
        y_orig, _ = librosa.load(original_audio_path, sr=sample_rate, duration=30)

        # 합성된 WAV를 임시 파일로 저장 후 로드
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(synthesized_wav_data)
            tmp_path = tmp.name

        y_synth, _ = librosa.load(tmp_path, sr=sample_rate)

        # 임시 파일 삭제
        import os
        try:
            os.unlink(tmp_path)
        except:
            pass

        # 길이 맞추기 (짧은 쪽에 맞춤)
        min_len = min(len(y_orig), len(y_synth))
        y_orig = y_orig[:min_len]
        y_synth = y_synth[:min_len]

        if min_len < sample_rate * 0.5:  # 0.5초 미만이면 너무 짧음
            return 0.0

        # 1. Spectral Similarity (Mel Spectrogram)
        mel_orig = librosa.feature.melspectrogram(y=y_orig, sr=sample_rate, n_mels=128)
        mel_synth = librosa.feature.melspectrogram(y=y_synth, sr=sample_rate, n_mels=128)

        # 길이 맞추기
        min_frames = min(mel_orig.shape[1], mel_synth.shape[1])
        mel_orig = mel_orig[:, :min_frames]
        mel_synth = mel_synth[:, :min_frames]

        # Cosine similarity
        mel_orig_flat = mel_orig.flatten()
        mel_synth_flat = mel_synth.flatten()
        spectral_sim = np.dot(mel_orig_flat, mel_synth_flat) / (
            np.linalg.norm(mel_orig_flat) * np.linalg.norm(mel_synth_flat) + 1e-8
        )

        # 2. Chroma Similarity (Pitch content)
        chroma_orig = librosa.feature.chroma_cqt(y=y_orig, sr=sample_rate)
        chroma_synth = librosa.feature.chroma_cqt(y=y_synth, sr=sample_rate)

        # 길이 맞추기
        min_frames = min(chroma_orig.shape[1], chroma_synth.shape[1])
        chroma_orig = chroma_orig[:, :min_frames]
        chroma_synth = chroma_synth[:, :min_frames]

        chroma_orig_flat = chroma_orig.flatten()
        chroma_synth_flat = chroma_synth.flatten()
        chroma_sim = np.dot(chroma_orig_flat, chroma_synth_flat) / (
            np.linalg.norm(chroma_orig_flat) * np.linalg.norm(chroma_synth_flat) + 1e-8
        )

        # 최종 스코어: 가중 평균 (Chroma가 피치 정확도에 더 중요)
        final_score = 0.4 * spectral_sim + 0.6 * chroma_sim

        return max(0.0, min(1.0, final_score))  # 0~1로 클리핑

    except Exception as e:
        print(f"[AutoMatcher] 유사도 계산 실패: {e}")
        return 0.0


def auto_match_parameters(original_audio_path, engine, raw_data, sample_rate=44100, progress_callback=None):
    """
    원본 음원과 MIDI를 비교해서 최적 파라미터를 자동 탐색

    Coarse-to-Fine Grid Search:
    - 1단계: 넓은 간격으로 빠르게 탐색
    - 2단계: 최적점 주변을 세밀하게 탐색

    Args:
        original_audio_path: 원본 음원 파일 경로
        engine: AegisEngine 인스턴스
        raw_data: engine.audio_to_midi()의 결과 (캐시된 분석 데이터)
        sample_rate: 샘플링 레이트
        progress_callback: 진행률 콜백 함수 (optional)

    Returns:
        dict: {
            'confidence_threshold': float,
            'min_note_duration_ms': int,
            'sustain_ms': int,
            'score': float  # 유사도 스코어 (0.0~1.0)
        }
    """
    print("[AutoMatcher] 🔍 자동 파라미터 매칭 시작...")

    # === 1단계: Coarse Search (넓은 간격) ===
    coarse_grid = {
        'confidence_threshold': [0.2, 0.4, 0.6],
        'min_note_duration_ms': [50, 150, 250],
        'sustain_ms': [100, 300, 500]
    }

    best_score = -1.0
    best_params = None
    total_iterations = (
        len(coarse_grid['confidence_threshold']) *
        len(coarse_grid['min_note_duration_ms']) *
        len(coarse_grid['sustain_ms'])
    )
    current_iteration = 0

    print(f"[AutoMatcher] 1단계: Coarse Grid Search ({total_iterations}개 조합)...")

    for conf in coarse_grid['confidence_threshold']:
        for min_dur in coarse_grid['min_note_duration_ms']:
            for sustain in coarse_grid['sustain_ms']:
                current_iteration += 1

                if progress_callback:
                    progress = current_iteration / total_iterations
                    progress_callback(progress, f"탐색 중... ({current_iteration}/{total_iterations})")

                try:
                    # MIDI 생성
                    midi_buffer = io.BytesIO()
                    engine.extract_events(
                        raw_data,
                        midi_buffer,
                        confidence_threshold=conf,
                        min_note_duration_ms=min_dur,
                        sustain_ms=sustain,
                        midi_program=27  # 기본 기타 패치
                    )

                    midi_buffer.seek(0)
                    midi_data = midi_buffer.read()

                    if len(midi_data) < 100:  # 너무 작으면 빈 MIDI
                        continue

                    # MIDI → WAV 합성
                    wav_data = synthesize_midi(midi_data, sample_rate=sample_rate)
                    if not wav_data:
                        continue

                    # 유사도 계산
                    score = _calculate_similarity(original_audio_path, wav_data, sample_rate)

                    print(f"  conf={conf:.1f}, dur={min_dur}, sus={sustain} → score={score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'confidence_threshold': conf,
                            'min_note_duration_ms': min_dur,
                            'sustain_ms': sustain
                        }

                except Exception as e:
                    print(f"  [AutoMatcher] 조합 실패 (conf={conf}, dur={min_dur}, sus={sustain}): {e}")
                    continue

    if not best_params:
        print("[AutoMatcher] ❌ 유효한 결과를 찾지 못했습니다.")
        return None

    print(f"[AutoMatcher] 1단계 최적값: {best_params}, score={best_score:.3f}")

    # === 2단계: Fine Search (최적점 주변 세밀 탐색) ===
    fine_grid = {
        'confidence_threshold': [
            max(0.1, best_params['confidence_threshold'] - 0.1),
            best_params['confidence_threshold'],
            min(0.9, best_params['confidence_threshold'] + 0.1)
        ],
        'min_note_duration_ms': [
            max(10, best_params['min_note_duration_ms'] - 50),
            best_params['min_note_duration_ms'],
            min(500, best_params['min_note_duration_ms'] + 50)
        ],
        'sustain_ms': [
            max(0, best_params['sustain_ms'] - 100),
            best_params['sustain_ms'],
            min(1000, best_params['sustain_ms'] + 100)
        ]
    }

    total_iterations = (
        len(fine_grid['confidence_threshold']) *
        len(fine_grid['min_note_duration_ms']) *
        len(fine_grid['sustain_ms'])
    )
    current_iteration = 0

    print(f"[AutoMatcher] 2단계: Fine Grid Search ({total_iterations}개 조합)...")

    for conf in fine_grid['confidence_threshold']:
        for min_dur in fine_grid['min_note_duration_ms']:
            for sustain in fine_grid['sustain_ms']:
                current_iteration += 1

                if progress_callback:
                    progress = current_iteration / total_iterations
                    progress_callback(progress, f"세밀 탐색 중... ({current_iteration}/{total_iterations})")

                try:
                    midi_buffer = io.BytesIO()
                    engine.extract_events(
                        raw_data,
                        midi_buffer,
                        confidence_threshold=conf,
                        min_note_duration_ms=int(min_dur),
                        sustain_ms=int(sustain),
                        midi_program=27
                    )

                    midi_buffer.seek(0)
                    midi_data = midi_buffer.read()

                    if len(midi_data) < 100:
                        continue

                    wav_data = synthesize_midi(midi_data, sample_rate=sample_rate)
                    if not wav_data:
                        continue

                    score = _calculate_similarity(original_audio_path, wav_data, sample_rate)

                    print(f"  conf={conf:.2f}, dur={min_dur}, sus={sustain} → score={score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'confidence_threshold': conf,
                            'min_note_duration_ms': int(min_dur),
                            'sustain_ms': int(sustain)
                        }

                except Exception as e:
                    print(f"  [AutoMatcher] 조합 실패: {e}")
                    continue

    print(f"[AutoMatcher] ✅ 최종 최적값: {best_params}, score={best_score:.3f}")

    return {
        **best_params,
        'score': best_score
    }
