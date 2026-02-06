"""
Aegis Engine - MIDI to Audio Synthesizer
FluidSynth 기반 MIDI → WAV 변환 모듈
"""

import subprocess
import tempfile
import os
from io import BytesIO
from pathlib import Path


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
