"""
Financial Technical Analysis for MIDI Transcription
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
주식 시장 기술적 분석을 MIDI 노트 추출에 적용

핵심 아이디어:
- 음정(pitch) = 주가(price)
- 시간(time) = 시간축(time)
- 변동성(volatility) = 노이즈/떨림
- 추세(trend) = 실제 멜로디

기존 Aegis (Median Filter 3-point):
    → 너무 단순, 복잡한 패턴 못 잡음

Financial Aegis (Bollinger + MACD + RSI):
    → 정교한 분석, 자동 아티큘레이션 감지
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import scipy.signal
from .financial_filters import FinancialNoiseFilters, multi_filter_consensus


class FinancialPitchAnalyzer:
    """
    주식 기술적 분석 기반 피치 분석기

    핵심 기능:
    1. SMA/EMA: 피치 트렌드 추출
    2. Bollinger Bands: 아티큘레이션 감지 (벤딩/비브라토)
    3. MACD: 슬라이드 감지
    4. RSI: Ghost note 필터링
    """

    def __init__(self, sr=22050, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length
        self.ms_per_frame = (hop_length / sr) * 1000

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Moving Averages (이동평균)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def simple_moving_average(self, data, window=5):
        """
        단순 이동평균 (SMA)

        주식: 가격의 평균 추세
        MIDI: 피치의 평균 트렌드 (떨림 제거)

        Args:
            data: f0 배열 (Hz)
            window: 윈도우 크기 (프레임 수)

        Returns:
            평활화된 피치
        """
        # NaN 처리
        valid_data = np.where(np.isnan(data), 0, data)

        # Convolution으로 이동평균
        kernel = np.ones(window) / window
        smoothed = np.convolve(valid_data, kernel, mode='same')

        # 원래 NaN 위치 복원
        smoothed[np.isnan(data)] = np.nan

        return smoothed

    def exponential_moving_average(self, data, span=5):
        """
        지수 이동평균 (EMA)

        주식: 최근 가격에 더 큰 가중치
        MIDI: 최근 피치에 더 큰 가중치 (빠른 반응)

        Args:
            data: f0 배열
            span: EMA 기간

        Returns:
            EMA 평활화된 피치
        """
        alpha = 2 / (span + 1)
        ema = np.full_like(data, np.nan)

        # 첫 유효 값 찾기
        first_valid = None
        for i, val in enumerate(data):
            if not np.isnan(val):
                ema[i] = val
                first_valid = i
                break

        if first_valid is None:
            return ema

        # EMA 계산
        for i in range(first_valid + 1, len(data)):
            if not np.isnan(data[i]):
                if np.isnan(ema[i-1]):
                    ema[i] = data[i]
                else:
                    ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

        return ema

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Bollinger Bands (볼린저 밴드)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def bollinger_bands(self, data, window=20, num_std=2):
        """
        볼린저 밴드

        주식: 가격의 정상 변동 범위
        MIDI: 피치의 정상 범위 → 이탈 = 특수 주법

        상단 밴드 초과 = 벤딩 업
        하단 밴드 미만 = 비정상 음정 (노이즈)
        밴드 내 = 정상 음정

        Args:
            data: f0 배열
            window: 이동평균 윈도우
            num_std: 표준편차 배수 (보통 2)

        Returns:
            (ma, upper_band, lower_band)
        """
        ma = self.simple_moving_average(data, window)

        # 롤링 표준편차
        std = np.full_like(data, np.nan)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            window_data = data[start:i+1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 1:
                std[i] = np.std(valid_data)

        upper = ma + (num_std * std)
        lower = ma - (num_std * std)

        return ma, upper, lower

    def detect_articulation_bollinger(self, f0, window=10, sensitivity=2.0):
        """
        볼린저 밴드로 아티큘레이션 감지

        Returns:
            각 프레임의 아티큘레이션 타입
            - 'normal': 정상 음정
            - 'bend': 벤딩 (상단 밴드 이탈)
            - 'vibrato': 비브라토 (밴드 경계 왔다갔다)
            - 'noise': 노이즈 (하단 밴드 이탈)
        """
        ma, upper, lower = self.bollinger_bands(f0, window, sensitivity)

        articulations = []
        prev_state = 'normal'
        vibrato_counter = 0

        for i in range(len(f0)):
            if np.isnan(f0[i]):
                articulations.append(None)
                continue

            # 밴드 위치 판단
            if f0[i] > upper[i]:
                state = 'above'
            elif f0[i] < lower[i]:
                state = 'below'
            else:
                state = 'normal'

            # 비브라토 감지 (밴드 경계 교차)
            if prev_state != state and prev_state != 'normal':
                vibrato_counter += 1
            else:
                vibrato_counter = 0

            # 아티큘레이션 분류
            if vibrato_counter >= 2:
                articulation = 'vibrato'
            elif state == 'above':
                articulation = 'bend'
            elif state == 'below':
                articulation = 'noise'
            else:
                articulation = 'normal'

            articulations.append(articulation)
            prev_state = state

        return articulations

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MACD (Moving Average Convergence Divergence)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def macd(self, data, fast=12, slow=26, signal=9):
        """
        MACD 지표

        주식: 추세 변화 및 모멘텀 감지
        MIDI: 피치 변화율 → 슬라이드 감지

        Args:
            data: f0 배열
            fast: 빠른 EMA 기간
            slow: 느린 EMA 기간
            signal: 시그널 라인 기간

        Returns:
            (macd_line, signal_line, histogram)
        """
        ema_fast = self.exponential_moving_average(data, span=fast)
        ema_slow = self.exponential_moving_average(data, span=slow)

        macd_line = ema_fast - ema_slow
        signal_line = self.exponential_moving_average(macd_line, span=signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def detect_slides_macd(self, f0, threshold=0.5):
        """
        MACD로 슬라이드 감지

        MACD > 0 & 증가 = 상승 슬라이드
        MACD < 0 & 감소 = 하강 슬라이드

        Args:
            f0: 피치 배열
            threshold: MACD 임계값 (semitones)

        Returns:
            각 프레임의 슬라이드 타입
        """
        import librosa

        # Hz → Semitone (상대 변화)
        f0_semitones = np.full_like(f0, np.nan)
        valid_mask = ~np.isnan(f0)
        if np.any(valid_mask):
            f0_semitones[valid_mask] = librosa.hz_to_midi(f0[valid_mask])

        macd_line, signal_line, histogram = self.macd(
            f0_semitones, fast=5, slow=20, signal=9
        )

        slides = []
        for i in range(len(macd_line)):
            if np.isnan(macd_line[i]):
                slides.append(None)
                continue

            # 슬라이드 판단
            if macd_line[i] > threshold and histogram[i] > 0:
                slides.append('slide_up')
            elif macd_line[i] < -threshold and histogram[i] < 0:
                slides.append('slide_down')
            else:
                slides.append('normal')

        return slides

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RSI (Relative Strength Index)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def rsi(self, data, period=14):
        """
        RSI 지표

        주식: 과매수/과매도 판단
        MIDI: 노트 과밀도 판단 → Ghost note 필터링

        RSI > 70: 과밀 (Ghost note 의심)
        RSI < 30: 정상

        Args:
            data: 노트 밀도 또는 피치 변화율
            period: RSI 기간

        Returns:
            RSI 값 배열
        """
        # 변화량 계산
        deltas = np.diff(data)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # 평균 gain/loss
        avg_gains = np.full(len(data), np.nan)
        avg_losses = np.full(len(data), np.nan)

        if len(gains) >= period:
            avg_gains[period] = np.mean(gains[:period])
            avg_losses[period] = np.mean(losses[:period])

            # EMA 방식으로 업데이트
            for i in range(period + 1, len(data)):
                avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
                avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period

        # RSI 계산
        rsi_values = np.full(len(data), 50.0)  # 기본값 50

        for i in range(period, len(data)):
            if avg_losses[i] == 0:
                rsi_values[i] = 100
            else:
                rs = avg_gains[i] / avg_losses[i]
                rsi_values[i] = 100 - (100 / (1 + rs))

        return rsi_values

    def filter_ghost_notes_rsi(self, note_events, rsi_threshold=70):
        """
        RSI로 Ghost Note 필터링

        과밀 구간의 노트 = Ghost note 가능성 높음

        Args:
            note_events: 노트 이벤트 리스트
            rsi_threshold: RSI 임계값 (기본 70)

        Returns:
            필터링된 노트 이벤트
        """
        if not note_events:
            return note_events

        # 시간축 노트 밀도 계산
        max_time = max(e['end'] for e in note_events)
        time_bins = np.linspace(0, max_time, int(max_time * 10))  # 100ms bins
        density = np.zeros(len(time_bins))

        for event in note_events:
            start_idx = int(event['start'] * 10)
            end_idx = int(event['end'] * 10)
            if start_idx < len(density):
                density[start_idx:min(end_idx, len(density))] += 1

        # RSI 계산
        rsi_values = self.rsi(density, period=14)

        # 필터링
        filtered_events = []
        for event in note_events:
            time_idx = int(event['start'] * 10)
            if time_idx < len(rsi_values):
                if rsi_values[time_idx] < rsi_threshold:
                    filtered_events.append(event)
            else:
                filtered_events.append(event)

        return filtered_events

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 통합 분석 (All-in-One)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def analyze_pitch_financial(self, f0, voiced_flag, use_advanced_filters=True):
        """
        주식 기법 통합 피치 분석

        Args:
            f0: 피치 배열
            voiced_flag: 유성음 플래그
            use_advanced_filters: 고급 필터 사용 (Kalman, Savgol 등)

        Returns:
            {
                'trend': 트렌드 라인 (평활화된 피치),
                'articulations': 아티큘레이션 타입,
                'slides': 슬라이드 감지,
                'confidence': 신뢰도 (볼린저 기반)
            }
        """
        # 1. Trend (고급 필터 vs 기본 EMA)
        if use_advanced_filters:
            # Multi-Filter Consensus (Savgol + Kalman + Holt-Winters)
            trend, filter_conf = multi_filter_consensus(
                f0,
                filters=['savgol', 'kalman', 'holt']
            )
        else:
            # 기본 EMA
            trend = self.exponential_moving_average(f0, span=5)
            filter_conf = np.ones_like(f0)

        # 2. Articulations (Bollinger)
        articulations = self.detect_articulation_bollinger(f0, window=10)

        # 3. Slides (MACD)
        slides = self.detect_slides_macd(f0, threshold=0.3)

        # 4. Confidence (Bollinger 기반)
        ma, upper, lower = self.bollinger_bands(f0, window=10)
        band_width = upper - lower

        confidence = np.zeros_like(f0)
        for i in range(len(f0)):
            if not np.isnan(f0[i]) and not np.isnan(band_width[i]):
                # 밴드가 좁을수록 신뢰도 높음
                if band_width[i] > 0:
                    confidence[i] = 1.0 / (1.0 + band_width[i])
                else:
                    confidence[i] = 1.0
            else:
                confidence[i] = 0.0

        return {
            'trend': trend,
            'articulations': articulations,
            'slides': slides,
            'confidence': confidence
        }
