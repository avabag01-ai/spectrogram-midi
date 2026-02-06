"""
화성 분석 (Harmonic Analysis) - 조성 기반 노트 필터링
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전체 노트를 먼저 분석해서 조성(Key)과 메이저/마이너를 파악
→ 스케일 밖 노트 = Ghost note 가능성 높음
→ 음악적 맥락 기반 필터링
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from collections import Counter
import librosa


class HarmonicAnalyzer:
    """
    조성 분석 및 음악 이론 기반 필터링
    """

    # 12개 음계 (Chromatic)
    CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # 메이저 스케일 (반음 간격)
    MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]  # W-W-H-W-W-W-H

    # 마이너 스케일 (Natural Minor)
    MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]  # W-H-W-W-H-W-W

    # 블루스 스케일 (일렉기타에 중요!)
    BLUES_INTERVALS = [0, 3, 5, 6, 7, 10]  # Minor Pentatonic + b5

    # 펜타토닉 마이너
    PENTA_MINOR_INTERVALS = [0, 3, 5, 7, 10]

    def __init__(self):
        pass

    @staticmethod
    def midi_to_pitch_class(midi_note):
        """
        MIDI 노트 → 음계 클래스 (0-11)
        예: C4 (60) → 0, C#4 (61) → 1, ..., B4 (71) → 11
        """
        return int(midi_note) % 12

    def detect_key(self, midi_notes, use_duration=False, durations=None):
        """
        조성 감지 (Key Detection)

        Args:
            midi_notes: MIDI 노트 번호 배열
            use_duration: 지속시간 가중치 사용 여부
            durations: 각 노트의 지속시간 (ms)

        Returns:
            {
                'key': 'C',
                'mode': 'major',  # 'major', 'minor', 'blues'
                'confidence': 0.85
            }
        """
        if len(midi_notes) == 0:
            return {'key': 'C', 'mode': 'major', 'confidence': 0.0}

        # Pitch class 히스토그램
        pitch_classes = [self.midi_to_pitch_class(n) for n in midi_notes]

        # 지속시간 가중치
        if use_duration and durations is not None:
            weights = durations
        else:
            weights = np.ones(len(pitch_classes))

        # 가중 히스토그램
        histogram = np.zeros(12)
        for pc, weight in zip(pitch_classes, weights):
            histogram[pc] += weight

        # 정규화
        histogram = histogram / (np.sum(histogram) + 1e-6)

        # 모든 키 & 모드 테스트
        best_key = 'C'
        best_mode = 'major'
        best_score = 0.0

        for root in range(12):
            # 메이저
            major_score = self._calculate_key_score(histogram, root, self.MAJOR_INTERVALS)
            if major_score > best_score:
                best_score = major_score
                best_key = self.CHROMATIC[root]
                best_mode = 'major'

            # 마이너
            minor_score = self._calculate_key_score(histogram, root, self.MINOR_INTERVALS)
            if minor_score > best_score:
                best_score = minor_score
                best_key = self.CHROMATIC[root]
                best_mode = 'minor'

            # 블루스 (일렉기타에서 중요!)
            blues_score = self._calculate_key_score(histogram, root, self.BLUES_INTERVALS)
            if blues_score > best_score:
                best_score = blues_score
                best_key = self.CHROMATIC[root]
                best_mode = 'blues'

        return {
            'key': best_key,
            'mode': best_mode,
            'confidence': best_score
        }

    def _calculate_key_score(self, histogram, root, intervals):
        """
        특정 키/모드에 대한 적합도 점수
        """
        score = 0.0
        for interval in intervals:
            pitch_class = (root + interval) % 12
            score += histogram[pitch_class]
        return score

    def get_scale_notes(self, key, mode):
        """
        조성에 해당하는 스케일 노트 (pitch class)

        Returns:
            List of pitch classes (0-11)
        """
        root = self.CHROMATIC.index(key)

        if mode == 'major':
            intervals = self.MAJOR_INTERVALS
        elif mode == 'minor':
            intervals = self.MINOR_INTERVALS
        elif mode == 'blues':
            intervals = self.BLUES_INTERVALS
        else:
            intervals = self.MAJOR_INTERVALS

        return [(root + interval) % 12 for interval in intervals]

    def filter_out_of_scale_notes(self, midi_notes, confidences, key_info, tolerance=1):
        """
        스케일 밖 노트 필터링

        Args:
            midi_notes: MIDI 노트 배열
            confidences: 각 노트의 신뢰도
            key_info: detect_key() 결과
            tolerance: 허용 범위 (반음 개수)
                       0 = 스케일만 허용
                       1 = ±1 반음 허용 (벤딩 고려)
                       2 = ±2 반음 허용 (크로마틱)

        Returns:
            (filtered_midi, filtered_confidence, out_of_scale_mask)
        """
        scale_notes = self.get_scale_notes(key_info['key'], key_info['mode'])

        out_of_scale = np.zeros(len(midi_notes), dtype=bool)

        for i, midi_note in enumerate(midi_notes):
            pc = self.midi_to_pitch_class(midi_note)

            # 스케일 노트와의 최소 거리
            min_distance = min(
                min(abs(pc - scale_pc), 12 - abs(pc - scale_pc))
                for scale_pc in scale_notes
            )

            if min_distance > tolerance:
                out_of_scale[i] = True

        # 필터링
        filtered_midi = midi_notes[~out_of_scale]
        filtered_confidence = confidences[~out_of_scale]

        return filtered_midi, filtered_confidence, out_of_scale

    def analyze_chord_progression(self, midi_notes, times, window_size=2000):
        """
        화음 진행 분석 (시간 구간별)

        Args:
            midi_notes: MIDI 노트 배열
            times: 각 노트의 시작 시간 (ms)
            window_size: 분석 윈도우 크기 (ms)

        Returns:
            List of {'time': ms, 'chord': 'C', 'quality': 'major'}
        """
        if len(midi_notes) == 0:
            return []

        chords = []
        max_time = np.max(times)

        for t in range(0, int(max_time), window_size):
            # 시간 구간 내 노트 추출
            mask = (times >= t) & (times < t + window_size)
            window_notes = midi_notes[mask]

            if len(window_notes) == 0:
                continue

            # 가장 많이 나온 노트 = 루트음 후보
            pitch_classes = [self.midi_to_pitch_class(n) for n in window_notes]
            most_common = Counter(pitch_classes).most_common(1)[0][0]

            # 간단한 메이저/마이너 판별
            third_major = (most_common + 4) % 12
            third_minor = (most_common + 3) % 12

            if third_major in pitch_classes:
                quality = 'major'
            elif third_minor in pitch_classes:
                quality = 'minor'
            else:
                quality = 'unknown'

            chords.append({
                'time': t,
                'chord': self.CHROMATIC[most_common],
                'quality': quality
            })

        return chords

    def adaptive_filter_by_context(self, midi_notes, times, confidences, key_info):
        """
        음악적 맥락 기반 적응형 필터링

        1. 화음 진행 분석
        2. 각 노트가 현재 화음과 맞는지 확인
        3. 맞지 않으면 confidence 페널티

        Returns:
            adjusted_confidences
        """
        chords = self.analyze_chord_progression(midi_notes, times)

        if len(chords) == 0:
            return confidences

        adjusted = confidences.copy()

        for i, (note, time) in enumerate(zip(midi_notes, times)):
            # 현재 시간대 화음 찾기
            current_chord = None
            for chord in chords:
                if chord['time'] <= time < chord['time'] + 2000:
                    current_chord = chord
                    break

            if current_chord is None:
                continue

            # 노트가 화음에 포함되는지
            pc = self.midi_to_pitch_class(note)
            chord_root = self.CHROMATIC.index(current_chord['chord'])

            # 화음 구성음 (트라이어드)
            if current_chord['quality'] == 'major':
                chord_tones = [chord_root, (chord_root + 4) % 12, (chord_root + 7) % 12]
            elif current_chord['quality'] == 'minor':
                chord_tones = [chord_root, (chord_root + 3) % 12, (chord_root + 7) % 12]
            else:
                continue

            # 화음 밖 노트 = 페널티
            if pc not in chord_tones:
                # 스케일 내 노트면 약한 페널티
                scale_notes = self.get_scale_notes(key_info['key'], key_info['mode'])
                if pc in scale_notes:
                    adjusted[i] *= 0.8  # 20% 페널티
                else:
                    adjusted[i] *= 0.5  # 50% 페널티

        return adjusted


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 통합 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_harmonic_filter(midi_notes, confidences, times=None, tolerance=1):
    """
    화성 분석 기반 필터링

    Args:
        midi_notes: MIDI 노트 배열
        confidences: 신뢰도 배열
        times: 시간 배열 (ms) - 옵션
        tolerance: 스케일 허용 범위 (0-2)

    Returns:
        {
            'key_info': dict,
            'filtered_midi': array,
            'filtered_confidence': array,
            'out_of_scale_mask': array
        }
    """
    analyzer = HarmonicAnalyzer()

    # 1. 조성 감지
    key_info = analyzer.detect_key(midi_notes)

    print(f"🎵 감지된 조성: {key_info['key']} {key_info['mode']} (신뢰도: {key_info['confidence']:.2f})")

    # 2. 스케일 밖 노트 필터링
    filtered_midi, filtered_conf, out_mask = analyzer.filter_out_of_scale_notes(
        midi_notes, confidences, key_info, tolerance
    )

    # 3. 맥락 기반 조정 (시간 정보 있을 때)
    if times is not None:
        filtered_conf = analyzer.adaptive_filter_by_context(
            filtered_midi, times[~out_mask], filtered_conf, key_info
        )

    return {
        'key_info': key_info,
        'filtered_midi': filtered_midi,
        'filtered_confidence': filtered_conf,
        'out_of_scale_mask': out_mask
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("🎵 Harmonic Analysis Test")
    print()

    analyzer = HarmonicAnalyzer()

    # 1. C Major 스케일 테스트
    c_major = np.array([60, 62, 64, 65, 67, 69, 71, 72])  # C D E F G A B C
    key_info = analyzer.detect_key(c_major)

    print("1. 조성 감지 테스트:")
    print(f"   입력: C Major 스케일")
    print(f"   결과: {key_info['key']} {key_info['mode']}")
    print(f"   신뢰도: {key_info['confidence']:.3f}")
    print()

    # 2. A Minor 테스트
    a_minor = np.array([69, 71, 72, 74, 76, 77, 79, 81])  # A B C D E F G A
    key_info2 = analyzer.detect_key(a_minor)

    print("2. 마이너 스케일 테스트:")
    print(f"   입력: A Minor 스케일")
    print(f"   결과: {key_info2['key']} {key_info2['mode']}")
    print(f"   신뢰도: {key_info2['confidence']:.3f}")
    print()

    # 3. 스케일 밖 노트 필터링
    noisy_c_major = np.array([60, 61, 62, 63, 64, 65, 67, 68, 69])  # C C# D D# E F G G# A
    confidences = np.ones(len(noisy_c_major))

    filtered, filtered_conf, out_mask = analyzer.filter_out_of_scale_notes(
        noisy_c_major, confidences, key_info, tolerance=0
    )

    print("3. 스케일 밖 노트 제거:")
    print(f"   입력: {noisy_c_major}")
    print(f"   출력: {filtered}")
    print(f"   제거됨: {np.sum(out_mask)}개 (C#, D#, G#)")
    print()

    # 4. 블루스 감지
    blues_notes = np.array([60, 63, 65, 66, 67, 70, 72])  # C Eb F F# G Bb C
    blues_key = analyzer.detect_key(blues_notes)

    print("4. 블루스 스케일 테스트:")
    print(f"   입력: C Blues 스케일")
    print(f"   결과: {blues_key['key']} {blues_key['mode']}")
    print(f"   신뢰도: {blues_key['confidence']:.3f}")
    print()

    print("✅ 화성 분석 작동!")
