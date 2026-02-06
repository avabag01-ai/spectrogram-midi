"""
주식 차트 노이즈 필터링 기법들
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
트레이더들이 쓰는 노이즈 제거 공식을 MIDI 피치에 적용

1. Savitzky-Golay Filter (다항 회귀 평활화)
2. Kalman Filter (예측 + 보정)
3. Holt-Winters (지수 평활)
4. ATR (Average True Range) - 변동성 측정
5. Ichimoku Cloud (일목균형표) - 추세 + 지지저항
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
import numpy as np
from scipy import signal


class FinancialNoiseFilters:
    """
    주식 트레이더들이 쓰는 노이즈 필터 모음
    """

    @staticmethod
    def savitzky_golay(data, window=11, polyorder=3):
        """
        Savitzky-Golay Filter (다항 회귀 평활화)

        주식: 가격 추세를 부드럽게 하면서 피크 보존
        MIDI: 피치 평활화하면서 벤딩 피크 유지

        장점: Median보다 부드럽고, 피크 왜곡 적음
        """
        # NaN 처리
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data

        filtered = np.full_like(data, np.nan)

        try:
            # NaN 구간 스킵
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) > window:
                valid_data = data[valid_mask]
                smoothed = signal.savgol_filter(
                    valid_data,
                    window_length=min(window, len(valid_data) if len(valid_data) % 2 == 1 else len(valid_data) - 1),
                    polyorder=polyorder,
                    mode='nearest'
                )
                filtered[valid_mask] = smoothed

        except Exception:
            # Fallback: 원본 리턴
            filtered = data

        return filtered

    @staticmethod
    def kalman_filter(data, process_variance=1e-5, measurement_variance=1e-1):
        """
        Kalman Filter (칼만 필터)

        주식: 예측 + 관측 융합으로 노이즈 제거
        MIDI: 이전 피치 경향 + 현재 측정값 융합

        장점: 매우 부드럽고, 급격한 변화도 추적
        """
        # NaN 처리
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data

        filtered = np.full_like(data, np.nan)

        # 초기 상태
        first_valid = np.where(valid_mask)[0][0]
        x_est = data[first_valid]  # 초기 추정값
        p_est = 1.0                # 초기 오차

        for i in range(len(data)):
            if not valid_mask[i]:
                filtered[i] = np.nan
                continue

            # 예측 단계
            x_pred = x_est
            p_pred = p_est + process_variance

            # 갱신 단계
            k = p_pred / (p_pred + measurement_variance)  # 칼만 이득
            x_est = x_pred + k * (data[i] - x_pred)
            p_est = (1 - k) * p_pred

            filtered[i] = x_est

        return filtered

    @staticmethod
    def holt_winters(data, alpha=0.3, beta=0.1):
        """
        Holt-Winters (홀트-윈터스 지수 평활)

        주식: 수준(level) + 추세(trend) 동시 추적
        MIDI: 피치 평균 + 피치 변화율 추적

        장점: 추세 변화에 빠르게 반응
        """
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data

        filtered = np.full_like(data, np.nan)

        # 초기값
        first_valid = np.where(valid_mask)[0]
        if len(first_valid) < 2:
            return data

        level = data[first_valid[0]]
        trend = data[first_valid[1]] - data[first_valid[0]]

        for i in range(len(data)):
            if not valid_mask[i]:
                filtered[i] = np.nan
                continue

            # 예측
            forecast = level + trend

            # 갱신
            level_new = alpha * data[i] + (1 - alpha) * forecast
            trend_new = beta * (level_new - level) + (1 - beta) * trend

            filtered[i] = level_new
            level = level_new
            trend = trend_new

        return filtered

    @staticmethod
    def atr_filter(data, window=14, threshold=2.0):
        """
        ATR (Average True Range) 기반 노이즈 필터

        주식: 변동성이 평균의 N배 초과하면 노이즈로 간주
        MIDI: 피치 변화가 평균 변화의 N배 초과 → Ghost note

        Returns:
            (filtered_data, noise_mask)
        """
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data, np.zeros_like(data, dtype=bool)

        # True Range 계산 (절대 변화량)
        tr = np.abs(np.diff(data))

        # ATR (이동평균)
        atr = np.full(len(data), np.nan)
        for i in range(window, len(tr)):
            atr[i] = np.nanmean(tr[max(0, i-window):i])

        # 노이즈 마스크 (변화량이 ATR * threshold 초과)
        noise_mask = np.zeros(len(data), dtype=bool)
        for i in range(1, len(data)):
            if not np.isnan(atr[i]) and not np.isnan(data[i]):
                if np.abs(data[i] - data[i-1]) > atr[i] * threshold:
                    noise_mask[i] = True

        # 필터링 (노이즈 구간 보간)
        filtered = data.copy()
        for i in range(len(noise_mask)):
            if noise_mask[i]:
                # 이전 값으로 대체
                filtered[i] = filtered[i-1] if i > 0 else data[i]

        return filtered, noise_mask

    @staticmethod
    def ichimoku_baseline(data, tenkan=9, kijun=26):
        """
        Ichimoku Cloud - Baseline (기준선)

        주식: (최고가 + 최저가) / 2의 이동평균
        MIDI: (최고 피치 + 최저 피치) / 2의 추세

        장점: 지지/저항 개념, 추세 명확
        """
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data

        # Tenkan-sen (전환선): 9일 중간값
        tenkan_line = np.full_like(data, np.nan)
        for i in range(tenkan, len(data)):
            window = data[max(0, i-tenkan):i]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                tenkan_line[i] = (np.max(valid_window) + np.min(valid_window)) / 2

        # Kijun-sen (기준선): 26일 중간값
        kijun_line = np.full_like(data, np.nan)
        for i in range(kijun, len(data)):
            window = data[max(0, i-kijun):i]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                kijun_line[i] = (np.max(valid_window) + np.min(valid_window)) / 2

        # Baseline = 기준선 (Kijun)
        return kijun_line

    @staticmethod
    def stochastic_oscillator(data, k_period=14, smooth=3):
        """
        Stochastic Oscillator (스토캐스틱)

        주식: 현재 가격이 최근 N일 범위에서 어디에 있는지 (0~100)
        MIDI: 현재 피치가 최근 범위에서 어디에 있는지
              → 급격한 점프 감지

        Returns:
            0~100 값 (50 근처 = 정상, 0/100 = 극단)
        """
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return np.full_like(data, 50.0)

        k_values = np.full_like(data, 50.0)

        for i in range(k_period, len(data)):
            window = data[max(0, i-k_period):i+1]
            valid_window = window[~np.isnan(window)]

            if len(valid_window) > 0:
                low = np.min(valid_window)
                high = np.max(valid_window)

                if high - low > 0:
                    k_values[i] = ((data[i] - low) / (high - low)) * 100

        # 평활화 (D 라인)
        d_values = np.full_like(k_values, 50.0)
        for i in range(smooth, len(k_values)):
            d_values[i] = np.mean(k_values[max(0, i-smooth):i+1])

        return d_values


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 통합 필터 (여러 필터 조합)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def multi_filter_consensus(data, filters=['savgol', 'kalman', 'holt']):
    """
    여러 필터의 합의(Consensus)

    주식: 여러 지표가 동시에 신호 → 강한 신호
    MIDI: 여러 필터가 일치하는 피치 → 높은 신뢰도

    Args:
        data: 피치 배열
        filters: 사용할 필터 목록

    Returns:
        (consensus_pitch, consensus_confidence)
    """
    results = []
    filter_obj = FinancialNoiseFilters()

    if 'savgol' in filters:
        results.append(filter_obj.savitzky_golay(data))

    if 'kalman' in filters:
        results.append(filter_obj.kalman_filter(data))

    if 'holt' in filters:
        results.append(filter_obj.holt_winters(data))

    if not results:
        return data, np.ones_like(data)

    # 합의: 중앙값 (Median of filters)
    stacked = np.array(results)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
        consensus = np.nanmedian(stacked, axis=0)

        # 신뢰도: 필터 간 표준편차 (작을수록 높음)
        std = np.nanstd(stacked, axis=0)

    confidence = 1.0 / (1.0 + std)  # 0~1

    return consensus, confidence


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("🎸 Financial Noise Filters Test")
    print()

    # 테스트 데이터 (노이즈가 섞인 피치)
    clean = np.array([261.6] * 5 + [293.7] * 5)  # C4 → D4
    noise = np.random.normal(0, 5, len(clean))
    noisy = clean + noise

    filters = FinancialNoiseFilters()

    print("1. Savitzky-Golay:")
    savgol = filters.savitzky_golay(noisy, window=5, polyorder=2)
    print(f"   Noise reduced: {np.std(noisy - clean):.2f} → {np.std(savgol - clean):.2f}")

    print()
    print("2. Kalman Filter:")
    kalman = filters.kalman_filter(noisy)
    print(f"   Noise reduced: {np.std(noisy - clean):.2f} → {np.std(kalman - clean):.2f}")

    print()
    print("3. Multi-Filter Consensus:")
    consensus, conf = multi_filter_consensus(noisy)
    print(f"   Noise reduced: {np.std(noisy - clean):.2f} → {np.std(consensus - clean):.2f}")
    print(f"   Avg confidence: {np.mean(conf):.3f}")

    print()
    print("✅ 모든 필터 작동!")
