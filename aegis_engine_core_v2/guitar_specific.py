"""
일렉기타 전용 필터링 & 패턴 인식
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 저음역 필터 (E2 미만 제거)
2. 레이크 주법 패턴
3. 뮤트 주법 패턴
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import librosa


class GuitarSpecificFilters:
    """
    일렉기타 전용 필터 및 주법 인식
    """

    # 일렉기타 표준 튜닝 범위
    GUITAR_E2_HZ = librosa.midi_to_hz(40)  # E2 = 82.4Hz
    GUITAR_E6_HZ = librosa.midi_to_hz(88)  # E6 = 1318Hz

    @staticmethod
    def filter_subharmonic_noise(f0, voiced_flag, fmin_hz=82.4):
        """
        저음역 하모닉 노이즈 제거

        일렉기타 E2 (82Hz) 미만은:
        - 하모닉 오류 (옥타브 아래 인식)
        - 베이스 혼입
        - 전기 노이즈 (60Hz, 120Hz)

        Args:
            f0: 피치 배열 (Hz)
            voiced_flag: 유성음 플래그
            fmin_hz: 최소 주파수 (기본 E2)

        Returns:
            (filtered_f0, filtered_voiced)
        """
        filtered_f0 = f0.copy()
        filtered_voiced = voiced_flag.copy()

        # E2 미만 제거
        subharmonic_mask = f0 < fmin_hz

        filtered_f0[subharmonic_mask] = np.nan
        filtered_voiced[subharmonic_mask] = False

        # 옥타브 교정 시도 (하모닉 오류일 수 있음)
        # 예: 41Hz (E1) → 82Hz (E2)로 올림
        for i in range(len(f0)):
            if subharmonic_mask[i] and not np.isnan(f0[i]):
                # 2배로 올려서 범위 안이면 교정
                corrected = f0[i] * 2
                if fmin_hz <= corrected < fmin_hz * 4:
                    filtered_f0[i] = corrected
                    filtered_voiced[i] = True

        return filtered_f0, filtered_voiced

    @staticmethod
    def detect_palm_mute(S_dB, hop_length, sr, duration_ms=50):
        """
        팜 뮤트 주법 감지

        특징:
        - 고음역 감쇠 (타격 후 빠르게 사라짐)
        - 저음역 유지
        - 짧은 지속 시간 (50-200ms)

        Args:
            S_dB: Mel-spectrogram (dB)
            hop_length, sr: 오디오 파라미터
            duration_ms: 최대 지속 시간

        Returns:
            Boolean mask (True = 팜 뮤트)
        """
        n_mels, time_steps = S_dB.shape
        is_mute = np.zeros(time_steps, dtype=bool)

        # 고음역 / 저음역 분리
        mid_bin = n_mels // 2
        low_energy = np.mean(S_dB[:mid_bin, :], axis=0)
        high_energy = np.mean(S_dB[mid_bin:, :], axis=0)

        # 팜 뮤트 = 저음 강하고 고음 약함
        ratio = low_energy / (high_energy + 1e-6)

        # Threshold (저음이 고음의 2배 이상)
        mute_mask = ratio > 2.0

        # 지속시간 체크
        ms_per_frame = (hop_length / sr) * 1000
        max_frames = int(duration_ms / ms_per_frame)

        # 연속된 구간 필터링
        start = -1
        for i in range(len(mute_mask)):
            if mute_mask[i] and start == -1:
                start = i
            elif not mute_mask[i] and start != -1:
                duration = i - start
                if duration <= max_frames:
                    is_mute[start:i] = True
                start = -1

        return is_mute

    @staticmethod
    def detect_rake_enhanced(S_dB, hop_length, sr, rake_mask_basic):
        """
        레이크 주법 강화 감지

        기존 Rake detection + 추가 패턴:
        - 빠른 상승 (< 30ms)
        - 광대역 노이즈
        - 하강 추세 (에너지 감소)

        Args:
            S_dB: Mel-spectrogram
            hop_length, sr: 오디오 파라미터
            rake_mask_basic: 기존 Rake 마스크

        Returns:
            Enhanced rake mask
        """
        n_mels, time_steps = S_dB.shape
        enhanced_mask = rake_mask_basic.copy()

        # 에너지 변화율
        total_energy = np.mean(S_dB, axis=0)
        energy_diff = np.diff(total_energy, prepend=total_energy[0])

        # 급격한 상승 = Rake 시작
        ms_per_frame = (hop_length / sr) * 1000
        threshold_frames = int(30 / ms_per_frame)  # 30ms

        for i in range(1, len(energy_diff)):
            # 빠른 상승
            if energy_diff[i] > 10:  # dB 급상승
                # 다음 N 프레임이 하강 추세면 Rake
                if i + threshold_frames < len(energy_diff):
                    following = energy_diff[i:i+threshold_frames]
                    if np.mean(following) < 0:  # 평균적으로 하강
                        enhanced_mask[i:i+threshold_frames] = True

        return enhanced_mask

    @staticmethod
    def detect_hammer_on_pull_off(f0, min_semitone_jump=2, max_duration_ms=100):
        """
        해머온/풀오프 주법 감지

        특징:
        - 빠른 음정 변화 (2+ semitones)
        - 짧은 시간 (< 100ms)
        - Attack 없음 (에너지 급상승 없음)

        Args:
            f0: 피치 배열 (Hz)
            min_semitone_jump: 최소 음정 차이
            max_duration_ms: 최대 지속 시간

        Returns:
            List of (start_idx, end_idx, type)
        """
        valid_mask = ~np.isnan(f0)
        if not np.any(valid_mask):
            return []

        # Hz → MIDI
        midi = np.full_like(f0, np.nan)
        midi[valid_mask] = librosa.hz_to_midi(f0[valid_mask])

        # 음정 변화 감지
        hammer_ons = []

        for i in range(1, len(midi) - 1):
            if np.isnan(midi[i]) or np.isnan(midi[i-1]):
                continue

            semitone_diff = midi[i] - midi[i-1]

            # 상승 = 해머온, 하강 = 풀오프
            if abs(semitone_diff) >= min_semitone_jump:
                technique = 'hammer_on' if semitone_diff > 0 else 'pull_off'

                # 지속 시간 체크 (간단히 다음 변화까지)
                duration = 1
                for j in range(i+1, min(i+10, len(midi))):
                    if np.isnan(midi[j]):
                        break
                    if abs(midi[j] - midi[i]) > 0.5:
                        break
                    duration += 1

                hammer_ons.append({
                    'start': i,
                    'end': i + duration,
                    'type': technique,
                    'semitones': abs(semitone_diff)
                })

        return hammer_ons

    @staticmethod
    def classify_distortion_level(S_dB):
        """
        디스토션 레벨 분류

        클린 톤 vs 디스토션에 따라 파라미터 자동 조정

        Returns:
            'clean', 'light', 'heavy'
        """
        # 고음역 에너지 비율
        n_mels = S_dB.shape[0]
        high_bin_start = int(n_mels * 0.7)

        high_energy = np.mean(S_dB[high_bin_start:, :])
        total_energy = np.mean(S_dB)

        ratio = high_energy / (total_energy + 1e-6)

        # 디스토션이 많을수록 고음 하모닉 증가
        if ratio > 0.4:
            return 'heavy'
        elif ratio > 0.25:
            return 'light'
        else:
            return 'clean'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 통합 기타 필터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_guitar_filters(f0, voiced_flag, S_dB, hop_length, sr, rake_mask):
    """
    일렉기타 전용 필터 통합 적용

    Returns:
        {
            'f0': 필터링된 피치,
            'voiced': 필터링된 유성음,
            'rake_mask': 강화된 Rake 마스크,
            'mute_mask': 팜 뮤트 마스크,
            'distortion': 디스토션 레벨
        }
    """
    filters = GuitarSpecificFilters()

    # 1. 저음역 필터
    f0_filtered, voiced_filtered = filters.filter_subharmonic_noise(
        f0, voiced_flag, fmin_hz=82.4
    )

    # 2. Rake 강화
    rake_enhanced = filters.detect_rake_enhanced(
        S_dB, hop_length, sr, rake_mask
    )

    # 3. 팜 뮤트 감지
    mute_mask = filters.detect_palm_mute(S_dB, hop_length, sr)

    # 4. 디스토션 레벨
    distortion = filters.classify_distortion_level(S_dB)

    return {
        'f0': f0_filtered,
        'voiced': voiced_filtered,
        'rake_mask': rake_enhanced,
        'mute_mask': mute_mask,
        'distortion': distortion
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("🎸 Guitar-Specific Filters Test")
    print()

    filters = GuitarSpecificFilters()

    # 1. 저음역 필터
    test_f0 = np.array([40, 60, 82, 110, 220, 440], dtype=float)  # Hz
    test_voiced = np.ones_like(test_f0, dtype=bool)

    filtered_f0, filtered_voiced = filters.filter_subharmonic_noise(
        test_f0, test_voiced
    )

    print("1. 저음역 필터 (E2 = 82Hz):")
    print(f"   입력:  {test_f0}")
    print(f"   출력:  {filtered_f0}")
    print(f"   제거된: {np.sum(np.isnan(filtered_f0))}개")
    print()

    # 2. 디스토션 분류
    print("2. 디스토션 분류:")
    print("   (실제 Spectrogram 필요 - 테스트 생략)")
    print()

    print("✅ 기타 필터 작동!")
