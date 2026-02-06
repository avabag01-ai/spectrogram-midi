"""
Aegis Engine - MIDI to Audio Synthesizer
FluidSynth 기반 MIDI → WAV 변환 모듈
"""

import subprocess
import tempfile
import os
import struct
import wave
import math
from io import BytesIO
from pathlib import Path

import numpy as np


class FluidSynthSynthesizer:
    """FluidSynth CLI를 사용한 MIDI → WAV 변환기"""

    def __init__(self, fluidsynth_path="/opt/homebrew/bin/fluidsynth"):
        """
        Args:
            fluidsynth_path: FluidSynth 실행 파일 경로
        """
        self.fluidsynth_path = fluidsynth_path
        self.soundfont = self._find_soundfont()

    def _find_soundfont(self):
        """시스템에서 SoundFont 파일 찾기"""
        # 우선순위 순서대로 SoundFont 경로 체크
        possible_paths = [
            "/opt/homebrew/Cellar/fluid-synth/2.5.2/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2",
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/soundfonts/default.sf2",
            "/opt/homebrew/share/soundfonts/default.sf2",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # 찾지 못한 경우 기본 경로 반환 (나중에 에러 처리)
        return possible_paths[0]

    def is_available(self):
        """FluidSynth가 사용 가능한지 확인"""
        try:
            result = subprocess.run(
                [self.fluidsynth_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def midi_to_wav(self, midi_data, sample_rate=44100):
        """
        MIDI 데이터를 WAV 오디오로 변환

        Args:
            midi_data: MIDI 파일의 바이트 데이터 (bytes 또는 BytesIO)
            sample_rate: 샘플링 레이트 (기본 44100Hz)

        Returns:
            bytes: WAV 파일 데이터 (성공 시)
            None: 변환 실패 시

        Raises:
            RuntimeError: FluidSynth 실행 실패 시
        """
        # BytesIO 객체인 경우 바이트로 변환
        if isinstance(midi_data, BytesIO):
            midi_data = midi_data.getvalue()

        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as midi_tmp:
            midi_tmp.write(midi_data)
            midi_path = midi_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
            wav_path = wav_tmp.name

        try:
            # FluidSynth 명령 실행
            # 중요: 옵션은 soundfont/midi 파일 경로보다 앞에 와야 함
            # -ni: non-interactive mode
            # -g: gain (볼륨 조절, 0.0~10.0)
            # -r: sample rate
            # -F: fast-render mode (output to file)
            cmd = [
                self.fluidsynth_path,
                "-ni",  # no interactive mode
                "-g", "0.8",  # 약간 낮춘 게인으로 클리핑 방지
                "-r", str(sample_rate),  # sample rate
                "-F", wav_path,  # output file (fast render mode)
                self.soundfont,  # soundfont 파일
                midi_path  # midi 파일
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # 최대 30초
                stdin=subprocess.DEVNULL  # stdin 차단
            )

            if result.returncode != 0:
                raise RuntimeError(f"FluidSynth 실행 실패: {result.stderr}")

            # WAV 파일 읽기
            with open(wav_path, "rb") as f:
                wav_data = f.read()

            return wav_data

        except subprocess.TimeoutExpired:
            raise RuntimeError("FluidSynth 실행 시간 초과 (30초)")

        except Exception as e:
            raise RuntimeError(f"MIDI 변환 중 오류 발생: {str(e)}")

        finally:
            # 임시 파일 삭제
            try:
                os.unlink(midi_path)
            except:
                pass
            try:
                os.unlink(wav_path)
            except:
                pass


# 전역 싱글톤 인스턴스
_synthesizer = None


def get_synthesizer():
    """FluidSynth Synthesizer 싱글톤 인스턴스 반환"""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = FluidSynthSynthesizer()
    return _synthesizer


def synthesize_midi(midi_data, sample_rate=44100):
    """
    편의 함수: MIDI 데이터를 WAV로 변환

    Args:
        midi_data: MIDI 바이트 데이터 또는 BytesIO
        sample_rate: 샘플링 레이트

    Returns:
        bytes: WAV 데이터 (성공 시)
        None: 실패 시
    """
    synth = get_synthesizer()

    if not synth.is_available():
        return None

    try:
        return synth.midi_to_wav(midi_data, sample_rate)
    except Exception as e:
        print(f"MIDI 합성 실패: {e}")
        return None


# =============================================================================
# ADSR Envelope Synthesizer - 소프트웨어 신디사이저
# =============================================================================

# 기타 전용 ADSR 프리셋
GUITAR_ADSR_PRESETS = {
    'nylon': {
        'attack_ms': 5, 'decay_ms': 80, 'sustain_level': 0.6,
        'release_ms': 200, 'waveform': 'triangle',
    },
    'steel': {
        'attack_ms': 3, 'decay_ms': 60, 'sustain_level': 0.5,
        'release_ms': 150, 'waveform': 'sawtooth',
    },
    'electric_clean': {
        'attack_ms': 5, 'decay_ms': 40, 'sustain_level': 0.7,
        'release_ms': 100, 'waveform': 'sawtooth',
    },
    'electric_overdrive': {
        'attack_ms': 2, 'decay_ms': 30, 'sustain_level': 0.8,
        'release_ms': 300, 'waveform': 'square',
    },
    'muted': {
        'attack_ms': 2, 'decay_ms': 20, 'sustain_level': 0.2,
        'release_ms': 30, 'waveform': 'sawtooth',
    },
}


class ADSRSynthesizer:
    """
    ADSR 엔벨로프 기반 소프트웨어 신디사이저.

    FluidSynth + SoundFont 방식의 한계를 보완하여,
    ADSR(Attack-Decay-Sustain-Release) 엔벨로프로 음색을 세밀하게
    제어할 수 있는 순수 소프트웨어 합성 엔진.

    numpy/scipy 기반이며 외부 신스 라이브러리 불필요.
    """

    def __init__(self, sr=44100):
        """
        ADSR 신디사이저 초기화.

        Args:
            sr: 샘플 레이트 (기본 44100Hz)
        """
        self.sr = sr

    # -----------------------------------------------------------------
    # 1) ADSR 엔벨로프 생성
    # -----------------------------------------------------------------
    def generate_envelope(self, num_samples, attack_ms=10, decay_ms=50,
                          sustain_level=0.7, release_ms=100):
        """
        ADSR 엔벨로프 배열 생성.

        Attack → Decay → Sustain → Release 구간을 연결한
        진폭(amplitude) 엔벨로프를 numpy 배열로 반환한다.

        Args:
            num_samples: 전체 샘플 수 (노트 길이)
            attack_ms: 어택 시간 (밀리초) - 소리가 최대 볼륨에 도달하는 시간
            decay_ms: 디케이 시간 (밀리초) - 최대 볼륨에서 서스테인 레벨로 떨어지는 시간
            sustain_level: 서스테인 레벨 (0.0 ~ 1.0) - 유지되는 볼륨
            release_ms: 릴리스 시간 (밀리초) - 노트 오프 후 소리가 사라지는 시간

        Returns:
            numpy.ndarray: 0.0~1.0 범위의 진폭 엔벨로프 (길이 = num_samples)
        """
        attack_samples = int(self.sr * attack_ms / 1000.0)
        decay_samples = int(self.sr * decay_ms / 1000.0)
        release_samples = int(self.sr * release_ms / 1000.0)

        # 서스테인 구간 = 전체에서 A/D/R을 뺀 나머지
        sustain_samples = max(0, num_samples - attack_samples - decay_samples - release_samples)

        # 각 구간 생성
        attack = np.linspace(0.0, 1.0, attack_samples, endpoint=False) if attack_samples > 0 else np.array([])
        decay = np.linspace(1.0, sustain_level, decay_samples, endpoint=False) if decay_samples > 0 else np.array([])
        sustain = np.full(sustain_samples, sustain_level) if sustain_samples > 0 else np.array([])
        release = np.linspace(sustain_level, 0.0, release_samples, endpoint=True) if release_samples > 0 else np.array([])

        envelope = np.concatenate([attack, decay, sustain, release])

        # 길이 보정 (반올림 오차 대응)
        if len(envelope) < num_samples:
            envelope = np.pad(envelope, (0, num_samples - len(envelope)), constant_values=0.0)
        elif len(envelope) > num_samples:
            envelope = envelope[:num_samples]

        return envelope

    # -----------------------------------------------------------------
    # 2) 오실레이터 (파형 생성)
    # -----------------------------------------------------------------
    def oscillator(self, freq, duration, waveform='sawtooth'):
        """
        기본 파형 오실레이터.

        주어진 주파수와 길이의 파형 신호를 생성한다.
        Sawtooth 파형이 기타 음색에 가장 적합하다.

        Args:
            freq: 주파수 (Hz)
            duration: 지속 시간 (초)
            waveform: 파형 종류 - 'sine', 'sawtooth', 'square', 'triangle'

        Returns:
            numpy.ndarray: -1.0~1.0 범위의 오디오 샘플 배열
        """
        num_samples = int(self.sr * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        if waveform == 'sine':
            # 정현파
            signal = np.sin(2 * np.pi * freq * t)

        elif waveform == 'sawtooth':
            # 톱니파 - 기타 음색에 적합
            # sawtooth: -1에서 1로 선형 증가 후 급격히 -1로 복귀
            phase = (freq * t) % 1.0
            signal = 2.0 * phase - 1.0

        elif waveform == 'square':
            # 사각파 - 오버드라이브/디스토션 느낌
            signal = np.sign(np.sin(2 * np.pi * freq * t))

        elif waveform == 'triangle':
            # 삼각파 - 부드러운 음색 (나일론 기타)
            phase = (freq * t) % 1.0
            signal = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0

        else:
            raise ValueError(f"지원하지 않는 파형: {waveform}. "
                             f"'sine', 'sawtooth', 'square', 'triangle' 중 선택.")

        return signal

    # -----------------------------------------------------------------
    # 3) 단일 노트 합성
    # -----------------------------------------------------------------
    def synthesize_note(self, freq, duration, velocity=100,
                        attack_ms=10, decay_ms=50, sustain_level=0.7,
                        release_ms=100, waveform='sawtooth', harmonics=True):
        """
        단일 노트를 ADSR 엔벨로프로 합성.

        기본 주파수에 오버톤(배음)을 추가하여 풍부한 음색을 구현하고,
        ADSR 엔벨로프와 벨로시티를 적용한다.

        Args:
            freq: 기본 주파수 (Hz)
            duration: 지속 시간 (초)
            velocity: MIDI 벨로시티 (0~127) - 음량 조절
            attack_ms: 어택 시간 (밀리초)
            decay_ms: 디케이 시간 (밀리초)
            sustain_level: 서스테인 레벨 (0.0~1.0)
            release_ms: 릴리스 시간 (밀리초)
            waveform: 파형 종류 ('sine', 'sawtooth', 'square', 'triangle')
            harmonics: True이면 배음 추가 (2~5차 배음, 감쇄 진폭)

        Returns:
            numpy.ndarray: 합성된 오디오 샘플 (float64)
        """
        num_samples = int(self.sr * duration)

        # 기본 파형 생성
        signal = self.oscillator(freq, duration, waveform)

        # 배음 추가 (2차, 3차, 4차, 5차 하모닉스 - 감쇄 진폭)
        if harmonics:
            harmonic_amplitudes = [0.5, 0.25, 0.125, 0.0625]  # 2nd ~ 5th 배음
            for i, amp in enumerate(harmonic_amplitudes):
                harmonic_num = i + 2
                harmonic_freq = freq * harmonic_num
                # 나이퀴스트 주파수를 초과하면 생략 (앨리어싱 방지)
                if harmonic_freq < self.sr / 2:
                    harmonic_signal = self.oscillator(harmonic_freq, duration, waveform)
                    signal = signal + amp * harmonic_signal

            # 배음 추가 후 정규화 (-1 ~ 1 범위 유지)
            peak = np.max(np.abs(signal))
            if peak > 0:
                signal = signal / peak

        # ADSR 엔벨로프 적용
        envelope = self.generate_envelope(
            num_samples,
            attack_ms=attack_ms,
            decay_ms=decay_ms,
            sustain_level=sustain_level,
            release_ms=release_ms,
        )
        signal = signal * envelope

        # 벨로시티 적용 (0~127 → 0.0~1.0)
        vel_scale = max(0.0, min(1.0, velocity / 127.0))
        signal = signal * vel_scale

        return signal

    # -----------------------------------------------------------------
    # 4) MIDI → WAV 변환 (ADSR 기반)
    # -----------------------------------------------------------------
    def midi_to_wav(self, midi_data, attack_ms=10, decay_ms=50,
                    sustain_level=0.7, release_ms=100, waveform='sawtooth'):
        """
        MIDI 데이터를 ADSR 엔벨로프 합성으로 WAV 오디오로 변환.

        mido 라이브러리로 MIDI를 파싱하고, 각 노트를 ADSR 파라미터로
        개별 합성한 뒤 모든 노트를 믹스다운하여 WAV 바이트를 반환한다.

        Args:
            midi_data: MIDI 파일 바이트 데이터 (bytes 또는 BytesIO)
            attack_ms: 어택 시간 (밀리초)
            decay_ms: 디케이 시간 (밀리초)
            sustain_level: 서스테인 레벨 (0.0~1.0)
            release_ms: 릴리스 시간 (밀리초)
            waveform: 파형 종류

        Returns:
            bytes: WAV 파일 데이터
        """
        import mido

        # BytesIO 처리
        if isinstance(midi_data, bytes):
            midi_data = BytesIO(midi_data)
        elif not isinstance(midi_data, BytesIO):
            midi_data = BytesIO(midi_data)

        mid = mido.MidiFile(file=midi_data)

        # 전체 길이 계산 (초 단위)
        total_seconds = mid.length
        if total_seconds <= 0:
            total_seconds = 10.0  # 안전장치

        # 릴리스 여유분 추가
        total_seconds += release_ms / 1000.0 + 0.5
        total_samples = int(self.sr * total_seconds)
        mixed = np.zeros(total_samples, dtype=np.float64)

        # MIDI 노트 번호 → 주파수 변환 (A4 = 440Hz 기준)
        def midi_note_to_freq(note):
            return 440.0 * (2.0 ** ((note - 69) / 12.0))

        # 모든 트랙의 노트 이벤트를 절대 시간으로 변환 후 합성
        for track in mid.tracks:
            current_time = 0.0  # 초 단위 절대 시간
            active_notes = {}   # {note_number: (start_time, velocity)}

            for msg in track:
                # delta time → 초 변환
                current_time += mido.tick2second(
                    msg.time, mid.ticks_per_beat,
                    self._get_tempo(mid, current_time)
                )

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = (current_time, msg.velocity)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time, velocity = active_notes.pop(msg.note)
                        duration = max(0.01, current_time - start_time)
                        freq = midi_note_to_freq(msg.note)

                        # 릴리스 포함한 실제 지속 시간
                        full_duration = duration + release_ms / 1000.0

                        note_signal = self.synthesize_note(
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

                        # 시작 위치에 노트 배치
                        start_sample = int(start_time * self.sr)
                        end_sample = start_sample + len(note_signal)

                        if end_sample > total_samples:
                            note_signal = note_signal[:total_samples - start_sample]
                            end_sample = total_samples

                        if start_sample < total_samples:
                            mixed[start_sample:end_sample] += note_signal

        # 마스터 정규화 (클리핑 방지)
        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak * 0.9  # 약간의 헤드룸 확보

        # float64 → int16 변환
        audio_int16 = np.clip(mixed * 32767, -32768, 32767).astype(np.int16)

        # WAV 바이트 생성 (wave + struct 모듈 사용)
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)           # 모노
            wf.setsampwidth(2)           # 16bit
            wf.setframerate(self.sr)
            wf.writeframes(audio_int16.tobytes())

        return wav_buffer.getvalue()

    def _get_tempo(self, mid, current_time_sec):
        """
        현재 시간에 해당하는 템포 값 반환.

        MIDI 파일의 set_tempo 메시지를 탐색하여 해당 시점의
        BPM(마이크로초/비트)을 반환한다.

        Args:
            mid: mido.MidiFile 객체
            current_time_sec: 현재 시간 (초) - 미래 확장용

        Returns:
            int: 템포 (마이크로초/비트, 기본값 500000 = 120 BPM)
        """
        tempo = 500000  # 기본값: 120 BPM
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break  # 첫 번째 템포 메시지 사용
        return tempo

    # -----------------------------------------------------------------
    # 5) 오디오 엔벨로프 분석 (원본 음색 매칭용)
    # -----------------------------------------------------------------
    def analyze_envelope(self, audio_data, sr=44100):
        """
        오디오 세그먼트에서 ADSR 유사 특성을 추출.

        RMS 에너지의 시간적 변화를 분석하여 Attack, Decay,
        Sustain, Release 파라미터를 추정한다.
        원본 오디오의 음색을 재현할 때 사용.

        Args:
            audio_data: numpy 배열 (float) 또는 int16 오디오 데이터
            sr: 샘플 레이트 (기본 44100)

        Returns:
            dict: {
                'attack_ms': float,   # 추정 어택 시간 (밀리초)
                'decay_ms': float,    # 추정 디케이 시간 (밀리초)
                'sustain_level': float, # 추정 서스테인 레벨 (0.0~1.0)
                'release_ms': float,  # 추정 릴리스 시간 (밀리초)
            }
        """
        # numpy 배열로 변환
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float64)

        # int16 → float 정규화
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float64) / 32768.0

        # 스테레오 → 모노
        if audio_data.ndim == 2:
            audio_data = np.mean(audio_data, axis=1)

        # RMS 에너지 계산 (프레임 단위)
        frame_size = int(sr * 0.005)  # 5ms 프레임
        hop_size = frame_size // 2
        num_frames = max(1, (len(audio_data) - frame_size) // hop_size + 1)

        rms = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_size
            end = min(start + frame_size, len(audio_data))
            frame = audio_data[start:end]
            rms[i] = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0.0

        if len(rms) == 0 or np.max(rms) == 0:
            # 무음 데이터 - 기본값 반환
            return {
                'attack_ms': 10.0,
                'decay_ms': 50.0,
                'sustain_level': 0.7,
                'release_ms': 100.0,
            }

        # RMS 정규화
        rms_norm = rms / np.max(rms)

        # --- Attack 추정: 0에서 피크까지의 시간 ---
        peak_frame = np.argmax(rms_norm)
        attack_frames = max(1, peak_frame)
        attack_ms = (attack_frames * hop_size / sr) * 1000.0

        # --- Sustain 레벨 추정: 피크 이후 중간 구간의 평균 RMS ---
        total_frames = len(rms_norm)
        if peak_frame < total_frames - 1:
            # 피크 이후 ~ 끝의 70% 지점까지를 서스테인 구간으로 간주
            sustain_start = peak_frame + max(1, int((total_frames - peak_frame) * 0.2))
            sustain_end = peak_frame + max(2, int((total_frames - peak_frame) * 0.7))
            sustain_end = min(sustain_end, total_frames)
            if sustain_start < sustain_end:
                sustain_level = float(np.mean(rms_norm[sustain_start:sustain_end]))
            else:
                sustain_level = 0.7
        else:
            sustain_level = 0.7

        sustain_level = max(0.05, min(1.0, sustain_level))

        # --- Decay 추정: 피크에서 서스테인 레벨에 처음 도달하는 시간 ---
        decay_frames = 0
        if peak_frame < total_frames - 1:
            for i in range(peak_frame, total_frames):
                if rms_norm[i] <= sustain_level * 1.05:  # 5% 오차 허용
                    decay_frames = i - peak_frame
                    break
            if decay_frames == 0:
                decay_frames = max(1, int((total_frames - peak_frame) * 0.15))
        else:
            decay_frames = 1

        decay_ms = (decay_frames * hop_size / sr) * 1000.0

        # --- Release 추정: 마지막으로 서스테인 레벨 아래로 떨어진 뒤 0에 도달하는 시간 ---
        release_frames = 0
        # 끝에서부터 탐색
        threshold = 0.05  # 거의 무음
        for i in range(total_frames - 1, -1, -1):
            if rms_norm[i] > threshold:
                release_frames = total_frames - 1 - i
                break

        if release_frames <= 0:
            release_frames = max(1, int(total_frames * 0.1))

        release_ms = (release_frames * hop_size / sr) * 1000.0

        # 합리적인 범위로 클램프
        attack_ms = max(1.0, min(500.0, attack_ms))
        decay_ms = max(1.0, min(1000.0, decay_ms))
        release_ms = max(5.0, min(2000.0, release_ms))

        return {
            'attack_ms': round(attack_ms, 1),
            'decay_ms': round(decay_ms, 1),
            'sustain_level': round(sustain_level, 3),
            'release_ms': round(release_ms, 1),
        }


# 전역 ADSR 싱글톤 인스턴스
_adsr_synthesizer = None


def get_adsr_synthesizer(sr=44100):
    """ADSRSynthesizer 싱글톤 인스턴스 반환"""
    global _adsr_synthesizer
    if _adsr_synthesizer is None or _adsr_synthesizer.sr != sr:
        _adsr_synthesizer = ADSRSynthesizer(sr=sr)
    return _adsr_synthesizer


def synthesize_midi_adsr(midi_data, preset='electric_clean', sample_rate=44100,
                         **adsr_overrides):
    """
    ADSR 소프트 신스로 MIDI 합성.

    기타 프리셋 기반으로 MIDI 데이터를 WAV 오디오로 변환한다.
    프리셋의 기본 ADSR 파라미터를 adsr_overrides로 개별 덮어쓸 수 있다.

    Args:
        midi_data: MIDI 바이트 데이터 (bytes 또는 BytesIO)
        preset: 기타 프리셋 이름 (기본 'electric_clean')
                사용 가능: 'nylon', 'steel', 'electric_clean',
                          'electric_overdrive', 'muted'
        sample_rate: 샘플 레이트 (기본 44100)
        **adsr_overrides: 프리셋 파라미터를 개별 덮어쓰기
                          예: attack_ms=20, sustain_level=0.5

    Returns:
        bytes: WAV 파일 데이터 (성공 시)
        None: 실패 시

    Examples:
        # 기본 electric_clean 프리셋으로 합성
        wav = synthesize_midi_adsr(midi_bytes)

        # nylon 프리셋 + 커스텀 릴리스
        wav = synthesize_midi_adsr(midi_bytes, preset='nylon', release_ms=300)

        # 원본 오디오에서 추출한 ADSR로 합성
        synth = get_adsr_synthesizer()
        params = synth.analyze_envelope(original_audio)
        wav = synthesize_midi_adsr(midi_bytes, **params)
    """
    synth = get_adsr_synthesizer(sr=sample_rate)

    # 프리셋 파라미터 가져오기
    if preset in GUITAR_ADSR_PRESETS:
        params = dict(GUITAR_ADSR_PRESETS[preset])
    else:
        # 알 수 없는 프리셋이면 electric_clean 기본값 사용
        print(f"경고: 알 수 없는 프리셋 '{preset}', 'electric_clean' 기본값 사용")
        params = dict(GUITAR_ADSR_PRESETS['electric_clean'])

    # 오버라이드 적용
    params.update(adsr_overrides)

    try:
        return synth.midi_to_wav(
            midi_data,
            attack_ms=params.get('attack_ms', 10),
            decay_ms=params.get('decay_ms', 50),
            sustain_level=params.get('sustain_level', 0.7),
            release_ms=params.get('release_ms', 100),
            waveform=params.get('waveform', 'sawtooth'),
        )
    except Exception as e:
        print(f"ADSR MIDI 합성 실패: {e}")
        return None
