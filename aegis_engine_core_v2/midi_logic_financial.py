"""
Financial MIDI Logic - 주식 기술적 분석 기반 MIDI 변환
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
기존 Aegis: Median Filter (3-point) + 하드코딩 threshold
Financial: Bollinger Bands + MACD + RSI + 자동 threshold
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import librosa
import scipy.signal
from mido import Message
from .financial_analysis import FinancialPitchAnalyzer
from .harmonic_analysis import HarmonicAnalyzer


def detect_articulations_financial(f0, start, end, analyzer):
    """
    Financial 방식으로 아티큘레이션 감지

    기존: slope + vibrato amplitude (하드코딩 threshold)
    개선: Bollinger Bands + MACD (자동 감지)

    Args:
        f0: 피치 배열
        start, end: 구간
        analyzer: FinancialPitchAnalyzer 인스턴스

    Returns:
        아티큘레이션 타입 ('bend', 'vibrato', 'slide', None)
    """
    if end <= start:
        return None

    slice_f0 = f0[start:end+1]
    slice_f0 = slice_f0[~np.isnan(slice_f0)]

    if len(slice_f0) < 3:
        return None

    # Bollinger 기반 아티큘레이션 (벤딩/비브라토)
    artic_list = analyzer.detect_articulation_bollinger(
        slice_f0,
        window=min(5, len(slice_f0)),
        sensitivity=1.5
    )

    # MACD 기반 슬라이드 감지
    slides_list = analyzer.detect_slides_macd(slice_f0, threshold=0.3)

    # 통합: Bollinger + MACD
    combined_counts = {}

    # Bollinger 결과
    for a in artic_list:
        if a and a != 'normal':
            combined_counts[a] = combined_counts.get(a, 0) + 1

    # MACD 결과 (슬라이드가 많으면 우선)
    slide_count = sum(1 for s in slides_list if s and s != 'normal')
    if slide_count >= 2:  # 최소 2프레임 이상
        combined_counts['slide'] = slide_count

    if not combined_counts:
        return None

    # 가장 많이 나온 타입 선택
    dominant = max(combined_counts.items(), key=lambda x: x[1])

    # 최소 빈도 체크 (30% 이상)
    total_frames = len(artic_list)
    if dominant[1] / total_frames >= 0.3:
        return dominant[0]

    return None


def adaptive_confidence_threshold(confidence_values, method='bollinger'):
    """
    자동 신뢰도 threshold 계산

    기존: 하드코딩 0.7
    개선: 데이터 기반 자동 계산

    Args:
        confidence_values: 신뢰도 배열
        method: 'bollinger' or 'percentile'

    Returns:
        최적 threshold
    """
    valid_conf = confidence_values[confidence_values > 0]

    if len(valid_conf) == 0:
        return 0.5

    if method == 'bollinger':
        # Bollinger 방식: 평균 - 1 std
        mean = np.mean(valid_conf)
        std = np.std(valid_conf)
        threshold = mean - std

        # 범위 제한
        threshold = np.clip(threshold, 0.3, 0.8)

    elif method == 'percentile':
        # 하위 30% 제거
        threshold = np.percentile(valid_conf, 30)
        threshold = np.clip(threshold, 0.3, 0.8)

    else:
        threshold = 0.5

    return threshold


def get_midi_events_financial(rake_mask, f0, voiced_flag, active_probs, rms, sr, hop_length,
                               confidence_threshold=None, **kwargs):
    """
    Financial Algorithm 기반 MIDI 이벤트 생성

    핵심 개선사항:
    1. Bollinger Bands로 트렌드 추출 (Median Filter 대체)
    2. 자동 confidence threshold (하드코딩 제거)
    3. MACD 기반 슬라이드 감지
    4. RSI 기반 Ghost note 필터링

    Args:
        rake_mask: Rake 노이즈 마스크
        f0: 피치 배열 (Hz)
        voiced_flag: 유성음 플래그
        active_probs: PYIN 신뢰도
        rms: RMS 에너지
        sr: 샘플레이트
        hop_length: 홉 길이
        confidence_threshold: 신뢰도 임계값 (None이면 자동)
        **kwargs: 추가 파라미터

    Returns:
        MIDI 이벤트 리스트
    """
    # 파라미터
    noise_gate_db = kwargs.get('noise_gate_db', -40)
    sustain_ms = kwargs.get('sustain_ms', 50)
    min_note_duration_ms = kwargs.get('min_note_duration_ms', 50)
    use_financial = kwargs.get('use_financial', True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 1: Financial Analysis (피치 정제 + 분석)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    analyzer = FinancialPitchAnalyzer(sr=sr, hop_length=hop_length)

    if use_financial:
        print("[Financial] 주식 기법 적용 중...")

        # NaN 처리 (PYIN은 무성음을 NaN으로 리턴)
        f0_clean = np.where(voiced_flag, f0, np.nan)

        # Financial 통합 분석
        analysis = analyzer.analyze_pitch_financial(f0_clean, voiced_flag)

        f0_smooth = analysis['trend']  # EMA 트렌드
        financial_confidence = analysis['confidence']  # Bollinger 기반 신뢰도
        articulations = analysis['articulations']
        slides = analysis['slides']

        # 신뢰도 통합 (PYIN + Bollinger)
        combined_confidence = active_probs * 0.5 + financial_confidence * 0.5

        # 자동 threshold 계산
        if confidence_threshold is None:
            confidence_threshold = adaptive_confidence_threshold(
                combined_confidence,
                method='bollinger'
            )
            print(f"[Financial] 자동 Threshold: {confidence_threshold:.3f}")

    else:
        # 기존 Aegis 방식 (fallback)
        print("[Financial] Fallback: 기존 Median Filter 사용")

        try:
            f0_smooth = librosa.util.softmask(f0, voiced_flag.astype(np.float64), margin=0.5)
            f0_smooth = scipy.signal.medfilt(f0_smooth, kernel_size=3)
        except Exception as e:
            print(f"[Financial] ⚠️ Smoothing 실패: {e}")
            f0_smooth = f0

        combined_confidence = active_probs
        articulations = [None] * len(f0)
        slides = [None] * len(f0)

        if confidence_threshold is None:
            confidence_threshold = 0.7

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 2: Event Extraction (노트 이벤트 추출)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    events = []
    current_event = None

    min_note_duration_frames = int((min_note_duration_ms / 1000.0) * sr / hop_length)
    sustain_frames = int((sustain_ms / 1000.0) * sr / hop_length)

    for t in range(len(f0_smooth)):
        freq = f0_smooth[t]
        is_voiced = voiced_flag[t] if not np.isnan(freq) else False
        confidence = combined_confidence[t]
        is_rake = rake_mask[t]
        energy = rms_db[t]

        # Noise Gate
        if energy < noise_gate_db:
            is_voiced = False

        # 유효 음정
        if is_voiced and not np.isnan(freq) and freq > 0 and not is_rake:
            midi_note = int(round(librosa.hz_to_midi(freq)))
            velocity = int(np.clip((energy + 80) * 1.5, 0, 127))

            # Financial 아티큘레이션 정보
            artic_type = articulations[t] if use_financial else None
            slide_type = slides[t] if use_financial else None

            if current_event is None:
                # 새 노트 시작
                current_event = {
                    'note': midi_note,
                    'start': t,
                    'end': t,
                    'confidence': confidence,
                    'velocity': velocity,
                    'track': 'main' if confidence >= confidence_threshold else 'safe',
                    'financial_artic': artic_type,
                    'financial_slide': slide_type
                }
            else:
                # 노트 지속 또는 변경
                if current_event['note'] == midi_note:
                    current_event['end'] = t
                    # 아티큘레이션 업데이트 (가장 빈번한 것)
                    if artic_type and artic_type != 'normal':
                        current_event['financial_artic'] = artic_type
                else:
                    # 노트 종료, 다음 노트 시작
                    if use_financial:
                        current_event['technique'] = current_event.get('financial_artic')
                    else:
                        current_event['technique'] = detect_articulations_financial(
                            f0_smooth,
                            current_event['start'],
                            current_event['end'],
                            analyzer
                        )

                    events.append(current_event)

                    current_event = {
                        'note': midi_note,
                        'start': t,
                        'end': t,
                        'confidence': confidence,
                        'velocity': velocity,
                        'track': 'main' if confidence >= confidence_threshold else 'safe',
                        'financial_artic': artic_type,
                        'financial_slide': slide_type
                    }
        else:
            # 무성음 구간
            if current_event is not None:
                if use_financial:
                    current_event['technique'] = current_event.get('financial_artic')
                else:
                    current_event['technique'] = detect_articulations_financial(
                        f0_smooth,
                        current_event['start'],
                        current_event['end'],
                        analyzer
                    )

                events.append(current_event)
                current_event = None

    # 마지막 노트 처리
    if current_event is not None:
        if use_financial:
            current_event['technique'] = current_event.get('financial_artic')
        events.append(current_event)

    if not events:
        return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 3: Post-Processing (후처리)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 최소 지속시간 필터
    events = [e for e in events if (e['end'] - e['start']) >= min_note_duration_frames]

    # Sustain 병합 (기존 로직 유지)
    if len(events) > 1:
        merged = []
        curr = events[0]

        for i in range(1, len(events)):
            next_evt = events[i]
            gap = next_evt['start'] - curr['end']

            # 같은 음정 + 짧은 갭 + 아티큘레이션 없음 = 병합
            if (next_evt['note'] == curr['note'] and
                gap <= sustain_frames and
                not curr.get('technique')):
                curr['end'] = next_evt['end']
            else:
                merged.append(curr)
                curr = next_evt

        merged.append(curr)
        events = merged

    # RSI Ghost Note 필터링 (Financial)
    if use_financial and len(events) > 10:
        print("[Financial] RSI Ghost Note 필터링...")
        events = analyzer.filter_ghost_notes_rsi(events, rsi_threshold=70)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 4: Harmonic Analysis (화성 분석 필터링) ✨ NEW!
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    harmonic_filter_enabled = kwargs.get('use_harmonic_filter', True)

    if use_financial and harmonic_filter_enabled and len(events) > 5:
        print("[Harmonic] 조성 분석 중...")
        harmonic_analyzer = HarmonicAnalyzer()

        # 이벤트에서 MIDI 노트 추출
        midi_notes = np.array([e['note'] for e in events])
        confidences = np.array([e['confidence'] for e in events])
        times = np.array([e['start'] * (hop_length / sr) * 1000 for e in events])

        # 조성 감지
        key_info = harmonic_analyzer.detect_key(midi_notes)
        print(f"[Harmonic] 감지된 조성: {key_info['key']} {key_info['mode']} (신뢰도: {key_info['confidence']:.2f})")

        # 스케일 밖 노트 필터링 (tolerance=1: 벤딩 고려)
        scale_tolerance = kwargs.get('harmonic_tolerance', 1)
        filtered_midi, filtered_conf, out_of_scale = harmonic_analyzer.filter_out_of_scale_notes(
            midi_notes, confidences, key_info, tolerance=scale_tolerance
        )

        removed_count = np.sum(out_of_scale)
        if removed_count > 0:
            print(f"[Harmonic] 스케일 밖 노트 제거: {removed_count}개")

            # 이벤트 필터링
            events_filtered = []
            for i, evt in enumerate(events):
                if not out_of_scale[i]:
                    # 신뢰도 업데이트 (화성 분석 반영)
                    evt['confidence'] = filtered_conf[len(events_filtered)]
                    evt['harmonic_valid'] = True
                    events_filtered.append(evt)
                else:
                    evt['harmonic_valid'] = False

            # 맥락 기반 신뢰도 조정
            if len(events_filtered) > 0:
                adjusted_conf = harmonic_analyzer.adaptive_filter_by_context(
                    np.array([e['note'] for e in events_filtered]),
                    np.array([e['start'] * (hop_length / sr) * 1000 for e in events_filtered]),
                    np.array([e['confidence'] for e in events_filtered]),
                    key_info
                )

                for i, evt in enumerate(events_filtered):
                    evt['confidence'] = adjusted_conf[i]
                    # 트랙 재분류 (신뢰도 변경됨)
                    evt['track'] = 'main' if adjusted_conf[i] >= confidence_threshold else 'safe'

            events = events_filtered
            events[0]['key_info'] = key_info  # 첫 이벤트에 조성 정보 저장

    print(f"[Financial] 최종 이벤트: {len(events)}개")

    return events
