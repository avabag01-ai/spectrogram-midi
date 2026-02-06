"""
Aegis Engine - Financial Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
주식 기술적 분석 기반 MIDI 변환 엔진

"로직 프로가 못 잡는 걸 주식으로 잡는다"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import librosa
import numpy as np
from mido import MidiFile, MidiTrack, Message
import os

# 기존 Core 모듈
from aegis_engine_core.stems import separate_stems
from aegis_engine_core.vision import detect_rake_patterns
from aegis_engine_core.tabs import generate_tabs, export_musicxml

# Financial Core v2
from aegis_engine_core_v2.midi_logic_financial import get_midi_events_financial
from aegis_engine_core_v2.guitar_specific import apply_guitar_filters


class AegisFinancialEngine:
    """
    Aegis Engine with Financial Technical Analysis

    핵심 개선:
    1. Bollinger Bands → 피치 트렌드 + 아티큘레이션
    2. MACD → 슬라이드 감지
    3. RSI → Ghost note 필터링
    4. 자동 confidence threshold
    """

    def __init__(self, sample_rate=22050, hop_length=512, n_fft=2048):
        self.sr = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.version = "2.0-Financial"

    def load_audio(self, file_path, start_time=0, end_time=None):
        """오디오 로드 + Spectrogram"""
        duration = (end_time - start_time) if end_time else None
        y, _ = librosa.load(file_path, sr=self.sr, offset=start_time, duration=duration)

        # Spectrogram for Rake detection
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        return y, S_dB

    def detect_rake_patterns(self, S_dB, rake_sensitivity=0.6):
        """Rake 패턴 감지"""
        return detect_rake_patterns(S_dB, self.hop_length, self.sr, rake_sensitivity)

    def pitch_tracking(self, y):
        """PYIN 피치 추출"""
        print("[Financial] PYIN 피치 추출 중...")

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('E2'),
            fmax=librosa.note_to_hz('C6'),
            sr=self.sr,
            hop_length=self.hop_length
        )

        return f0, voiced_flag, voiced_probs

    def audio_to_midi_financial(self, input_wav, output_mid, **kwargs):
        """
        Financial Algorithm 기반 MIDI 변환

        Args:
            input_wav: 입력 오디오
            output_mid: 출력 MIDI
            **kwargs:
                - confidence_threshold: 신뢰도 (None=자동)
                - rake_sensitivity: Rake 감지 민감도
                - noise_gate_db: 노이즈 게이트
                - min_note_duration_ms: 최소 노트 길이
                - use_financial: Financial 알고리즘 사용 여부

        Returns:
            MIDI 파일 경로
        """
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🎸 Aegis Financial Engine v{self.version}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print()

        # 파라미터
        confidence_threshold = kwargs.get('confidence_threshold', None)
        rake_sensitivity = kwargs.get('rake_sensitivity', 0.6)
        use_financial = kwargs.get('use_financial', True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Phase 1: Audio Loading
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("[1/5] 오디오 로딩...")
        y, S_dB = self.load_audio(input_wav)
        print(f"      길이: {len(y)/self.sr:.1f}초")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Phase 2: Rake Detection (Vision AI)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("[2/5] Rake 패턴 감지 (Vision AI)...")
        rake_mask = self.detect_rake_patterns(S_dB, rake_sensitivity)
        rake_count = np.sum(rake_mask)
        print(f"      감지: {rake_count}개 프레임")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Phase 3: Pitch Tracking (PYIN)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("[3/5] 피치 추출 (PYIN)...")
        f0, voiced_flag, voiced_probs = self.pitch_tracking(y)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Phase 3.5: Guitar-Specific Filters ✨ NEW!
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        use_guitar_filters = kwargs.get('use_guitar_filters', True)

        if use_guitar_filters:
            print("[3.5/5] Guitar-Specific 필터...")
            guitar_result = apply_guitar_filters(
                f0, voiced_flag, S_dB, self.hop_length, self.sr, rake_mask
            )

            # 필터링된 결과 적용
            f0 = guitar_result['f0']
            voiced_flag = guitar_result['voiced']
            rake_mask = guitar_result['rake_mask']  # 강화된 Rake
            mute_mask = guitar_result['mute_mask']
            distortion_level = guitar_result['distortion']

            print(f"      디스토션: {distortion_level}")
            print(f"      Mute 감지: {np.sum(mute_mask)}개 프레임")

            # Mute 구간도 제거
            voiced_flag = voiced_flag & ~mute_mask

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Phase 4: Financial Analysis + MIDI Events
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("[4/5] Financial Analysis...")
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        # kwargs에서 중복 제거
        kwargs_filtered = {k: v for k, v in kwargs.items()
                          if k not in ['confidence_threshold', 'rake_sensitivity', 'use_financial']}

        events = get_midi_events_financial(
            rake_mask=rake_mask,
            f0=f0,
            voiced_flag=voiced_flag,
            active_probs=voiced_probs,
            rms=rms,
            sr=self.sr,
            hop_length=self.hop_length,
            confidence_threshold=confidence_threshold,
            use_financial=use_financial,
            **kwargs_filtered
        )

        if not events:
            print("⚠️  노트가 감지되지 않았습니다!")
            return None

        # 트랙 분리 통계
        main_count = sum(1 for e in events if e['track'] == 'main')
        safe_count = sum(1 for e in events if e['track'] == 'safe')

        print(f"      Main Track: {main_count}개 ({main_count/(main_count+safe_count)*100:.1f}%)")
        print(f"      Safe Track: {safe_count}개 ({safe_count/(main_count+safe_count)*100:.1f}%)")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Phase 5: MIDI Export
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("[5/5] MIDI 생성...")

        mid = MidiFile()
        track_main = MidiTrack()
        track_safe = MidiTrack()

        mid.tracks.append(track_main)
        mid.tracks.append(track_safe)

        # 트랙 이름 (MetaMessage 사용)
        from mido import MetaMessage
        track_main.append(MetaMessage('track_name', name='Aegis Financial - Main', time=0))
        track_safe.append(MetaMessage('track_name', name='Aegis Financial - Safe', time=0))

        # 이벤트 변환
        ticks_per_beat = mid.ticks_per_beat
        ms_per_tick = 500 / ticks_per_beat  # 120 BPM 기준

        last_time_main = 0
        last_time_safe = 0

        for evt in events:
            track = track_main if evt['track'] == 'main' else track_safe
            last_time = last_time_main if evt['track'] == 'main' else last_time_safe

            # 절대 시간 → 상대 시간 (delta time)
            ms_per_frame = (self.hop_length / self.sr) * 1000
            start_ms = evt['start'] * ms_per_frame
            duration_ms = (evt['end'] - evt['start']) * ms_per_frame

            start_ticks = int(start_ms / ms_per_tick)
            duration_ticks = int(duration_ms / ms_per_tick)

            delta_start = start_ticks - last_time

            # Note On
            track.append(Message(
                'note_on',
                note=evt['note'],
                velocity=evt['velocity'],
                time=delta_start
            ))

            # Note Off
            track.append(Message(
                'note_off',
                note=evt['note'],
                velocity=0,
                time=duration_ticks
            ))

            # 시간 업데이트
            if evt['track'] == 'main':
                last_time_main = start_ticks + duration_ticks
            else:
                last_time_safe = start_ticks + duration_ticks

        mid.save(output_mid)

        print()
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"✅ 완료: {output_mid}")
        print(f"   Total: {len(events)}개 노트")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return output_mid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 간단 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys

    print("Aegis Financial Engine - Quick Test")
    print()

    engine = AegisFinancialEngine()

    # 기본값
    test_input = "synthetic_guitar_test.wav"
    test_output = "test_financial_output.mid"

    # 커맨드 라인 인자 확인
    if len(sys.argv) >= 3:
        test_input = sys.argv[1]
        test_output = sys.argv[2]
        print(f"📍 Input: {test_input}")
        print(f"📍 Output: {test_output}")
        print("-" * 30)

    if os.path.exists(test_input):
        engine.audio_to_midi_financial(
            test_input,
            test_output,
            confidence_threshold=None,  # 자동
            rake_sensitivity=0.6,
            use_financial=True
        )
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {test_input}")
        if len(sys.argv) < 3:
            print("Usage: python3 aegis_engine_financial.py <input_audio> <output_midi>")
