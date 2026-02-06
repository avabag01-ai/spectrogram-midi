"""
Aegis Engine - Reverse Analyzer
MIDI → 합성 음원 → 다시 MIDI 변환 → 원본 MIDI와 비교
"""

import numpy as np
import tempfile
import os
import io
import mido
from aegis_engine_core.synthesizer import synthesize_midi


def _extract_notes_from_midi(midi_data):
    """
    MIDI 바이트 데이터에서 노트 정보 추출

    Returns:
        list: [{'pitch': int, 'start_time': float, 'end_time': float, 'velocity': int}, ...]
    """
    try:
        # BytesIO로 MIDI 파싱
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
                    # Note On
                    time_sec = mido.tick2second(current_time, ticks_per_beat, tempo)
                    active_notes[msg.note] = (time_sec, msg.velocity)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Note Off
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
        print(f"[ReverseAnalyzer] MIDI 노트 추출 실패: {e}")
        return []


def _compare_note_lists(original_notes, reversed_notes, time_tolerance=0.1, pitch_tolerance=1):
    """
    두 노트 리스트의 유사도 계산

    Args:
        original_notes: 원본 MIDI 노트 리스트
        reversed_notes: 역변환 MIDI 노트 리스트
        time_tolerance: 시간 허용 오차 (초)
        pitch_tolerance: 피치 허용 오차 (반음)

    Returns:
        dict: {
            'note_accuracy': float,  # 노트 일치율 (0.0~1.0)
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
            # 피치 차이
            pitch_diff = abs(orig['pitch'] - rev['pitch'])

            # 시작 시간 차이
            time_diff = abs(orig['start_time'] - rev['start_time'])

            # 종합 거리 (정규화)
            distance = (pitch_diff / 12.0) + time_diff

            if distance < best_distance:
                best_distance = distance
                best_match = rev

        # 매칭 판단
        if best_match:
            pitch_diff = abs(orig['pitch'] - best_match['pitch'])
            time_diff = abs(orig['start_time'] - best_match['start_time'])

            if pitch_diff <= pitch_tolerance and time_diff <= time_tolerance:
                matched_count += 1

            pitch_errors.append(pitch_diff)
            timing_errors.append(time_diff)

    # 정확도 계산
    note_accuracy = matched_count / len(original_notes)

    # 피치 정확도: 평균 오차를 반전 (오차 작을수록 높음)
    avg_pitch_error = np.mean(pitch_errors) if pitch_errors else 12.0
    pitch_accuracy = max(0.0, 1.0 - (avg_pitch_error / 12.0))  # 1옥타브 이내 오차 기준

    # 타이밍 정확도: 평균 오차를 반전
    avg_timing_error = np.mean(timing_errors) if timing_errors else 1.0
    timing_accuracy = max(0.0, 1.0 - (avg_timing_error / 0.5))  # 0.5초 이내 오차 기준

    return {
        'note_accuracy': note_accuracy,
        'pitch_accuracy': pitch_accuracy,
        'timing_accuracy': timing_accuracy
    }


def reverse_analysis(midi_data, engine, sample_rate=44100):
    """
    역변환 분석: MIDI → 합성 음원 → 다시 MIDI 변환 → 비교

    Args:
        midi_data: 원본 MIDI 바이트
        engine: AegisEngine 인스턴스
        sample_rate: 샘플링 레이트

    Returns:
        dict: {
            'original_notes': int,     # 원본 MIDI 노트 수
            'reversed_notes': int,      # 역변환 MIDI 노트 수
            'note_accuracy': float,     # 노트 일치율 (0.0~1.0)
            'pitch_accuracy': float,    # 피치 정확도 (0.0~1.0)
            'timing_accuracy': float,   # 타이밍 정확도 (0.0~1.0)
            'reversed_midi': bytes,     # 역변환 MIDI 바이트
            'reversed_events': list     # 역변환 이벤트 리스트
        }
    """
    print("[ReverseAnalyzer] 🔄 역변환 분석 시작...")

    try:
        # 1. 원본 MIDI 노트 추출
        print("[ReverseAnalyzer] 1/4 원본 MIDI 노트 추출 중...")
        original_notes = _extract_notes_from_midi(midi_data)
        print(f"  원본 노트 수: {len(original_notes)}")

        if not original_notes:
            print("[ReverseAnalyzer] ❌ 원본 MIDI에 노트가 없습니다.")
            return None

        # 2. MIDI → WAV 합성
        print("[ReverseAnalyzer] 2/4 MIDI → WAV 합성 중...")
        wav_data = synthesize_midi(midi_data, sample_rate=sample_rate)

        if not wav_data:
            print("[ReverseAnalyzer] ❌ MIDI 합성 실패")
            return None

        # 3. WAV를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav.write(wav_data)
            tmp_wav_path = tmp_wav.name

        # 4. WAV → MIDI 역변환
        print("[ReverseAnalyzer] 3/4 WAV → MIDI 역변환 중...")
        raw_data = engine.audio_to_midi(tmp_wav_path, None, turbo_mode=False)

        if not raw_data:
            print("[ReverseAnalyzer] ❌ 역변환 실패")
            os.unlink(tmp_wav_path)
            return None

        # 5. MIDI 이벤트 추출
        print("[ReverseAnalyzer] 4/4 MIDI 이벤트 추출 중...")
        reversed_midi_buffer = io.BytesIO()
        reversed_events = engine.extract_events(
            raw_data,
            reversed_midi_buffer,
            confidence_threshold=0.3,  # 역변환은 낮은 임계값 사용
            min_note_duration_ms=50,
            sustain_ms=200,
            midi_program=27
        )

        reversed_midi_buffer.seek(0)
        reversed_midi_data = reversed_midi_buffer.read()

        # 임시 파일 삭제
        try:
            os.unlink(tmp_wav_path)
        except:
            pass

        # 6. 역변환 MIDI 노트 추출
        reversed_notes = _extract_notes_from_midi(reversed_midi_data)
        print(f"  역변환 노트 수: {len(reversed_notes)}")

        # 7. 비교 분석
        print("[ReverseAnalyzer] 비교 분석 중...")
        comparison = _compare_note_lists(original_notes, reversed_notes)

        result = {
            'original_notes': len(original_notes),
            'reversed_notes': len(reversed_notes),
            'note_accuracy': comparison['note_accuracy'],
            'pitch_accuracy': comparison['pitch_accuracy'],
            'timing_accuracy': comparison['timing_accuracy'],
            'reversed_midi': reversed_midi_data,
            'reversed_events': reversed_events
        }

        print(f"[ReverseAnalyzer] ✅ 분석 완료!")
        print(f"  노트 일치율: {result['note_accuracy']:.1%}")
        print(f"  피치 정확도: {result['pitch_accuracy']:.1%}")
        print(f"  타이밍 정확도: {result['timing_accuracy']:.1%}")

        return result

    except Exception as e:
        print(f"[ReverseAnalyzer] ❌ 역변환 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
