"""
Aegis Engine - Guitar Fret Filter
운지 기반 물리적 노이즈 필터링

MIDI 노트를 기타 프렛보드에 매핑하고,
연속된 노트 간 물리적으로 불가능한 이동을 감지하여 노이즈를 제거한다.
"""

# 표준 튜닝 (high E → low E), MIDI 노트 번호
STANDARD_TUNING = [64, 59, 55, 50, 45, 40]  # E4, B3, G3, D3, A2, E2
STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E']
MAX_FRETS = 24

# 기타 연주 가능 범위: MIDI 40 (E2 open) ~ 88 (E4 + 24 = E6)
GUITAR_MIDI_MIN = 40
GUITAR_MIDI_MAX = 88


def midi_to_fret_positions(midi_note, tuning=None):
    """
    MIDI 노트를 가능한 (string_index, fret) 위치 목록으로 변환.

    Args:
        midi_note: MIDI 노트 번호
        tuning: 튜닝 배열 (기본: 표준 EADGBE)

    Returns:
        list of (string_index, fret) 튜플. string_index 0=high E, 5=low E
    """
    if tuning is None:
        tuning = STANDARD_TUNING

    positions = []
    for s_idx, open_pitch in enumerate(tuning):
        fret = midi_note - open_pitch
        if 0 <= fret <= MAX_FRETS:
            positions.append((s_idx, fret))
    return positions


def min_fret_distance(positions_a, positions_b):
    """
    두 노트의 모든 가능한 포지션 조합 중 최소 프렛 이동 거리를 계산.

    같은 줄이든 다른 줄이든 프렛 거리만 측정.
    (줄 변경 자체는 물리적 이동 비용이 거의 없음)

    Args:
        positions_a: 첫 번째 노트의 (string, fret) 리스트
        positions_b: 두 번째 노트의 (string, fret) 리스트

    Returns:
        (min_distance, best_pos_a, best_pos_b)
    """
    if not positions_a or not positions_b:
        return (999, None, None)

    best_dist = 999
    best_a = positions_a[0]
    best_b = positions_b[0]

    for pa in positions_a:
        for pb in positions_b:
            # 오픈 스트링(fret=0)은 손이 자유로우므로 거리 0 취급
            if pa[1] == 0 or pb[1] == 0:
                dist = 0
            else:
                dist = abs(pa[1] - pb[1])

            if dist < best_dist:
                best_dist = dist
                best_a = pa
                best_b = pb

    return (best_dist, best_a, best_b)


def _removal_score(event, sr, hop_length):
    """
    노트 제거 우선순위 점수 계산. 낮을수록 제거 대상.

    긴 노트, 높은 confidence, 테크닉 있는 노트는 보호.
    """
    duration_sec = (event['end'] - event['start']) * hop_length / sr
    confidence = event.get('confidence', 0.5)
    velocity = event.get('velocity', 64) / 127.0

    score = 0.0
    score += duration_sec * 10.0
    score += confidence * 5.0
    score += velocity * 2.0

    technique = event.get('technique')
    if technique in ('bend', 'vibrato', 'slide', 'hammer_on', 'pull_off'):
        score += 3.0

    return score


def apply_fret_filter(events, sr=44100, hop_length=512,
                      max_fret_speed=40.0,
                      protect_long_notes_ms=200.0,
                      min_confidence_protect=0.85):
    """
    운지 기반 물리적 노이즈 필터.

    연속된 노트 쌍의 프렛 거리와 시간 갭을 비교하여
    인간이 물리적으로 불가능한 이동을 감지, 노이즈를 제거한다.

    Args:
        events: 노트 이벤트 리스트 [{note, start, end, velocity, confidence, technique, ...}]
        sr: 샘플 레이트
        hop_length: 프레임 hop 길이
        max_fret_speed: 최대 허용 프렛 이동 속도 (frets/sec)
        protect_long_notes_ms: 이 길이 이상의 노트는 절대 제거 안함
        min_confidence_protect: 이 confidence 이상의 노트는 절대 제거 안함

    Returns:
        (filtered_events, report_dict)
    """
    if not events or len(events) < 2:
        return (list(events), _empty_report(len(events)))

    protect_long_frames = int((protect_long_notes_ms / 1000.0) * sr / hop_length)

    # 1. 각 노트의 프렛 포지션 계산
    positions_map = []
    for evt in events:
        positions_map.append(midi_to_fret_positions(evt['note']))

    # 2. 연속 쌍 분석 → 제거 후보 수집
    remove_indices = set()
    removed_details = []

    for i in range(len(events) - 1):
        curr = events[i]
        nxt = events[i + 1]

        # 동시 발음(코드)은 건너뜀
        if abs(nxt['start'] - curr['start']) < 2:
            continue

        # 기타 범위 밖 노트는 무조건 제거
        if curr['note'] < GUITAR_MIDI_MIN or curr['note'] > GUITAR_MIDI_MAX:
            if i not in remove_indices:
                remove_indices.add(i)
                removed_details.append({
                    'index': i, 'note': curr['note'],
                    'start': curr['start'], 'end': curr['end'],
                    'reason': 'out_of_guitar_range'
                })
            continue
        if nxt['note'] < GUITAR_MIDI_MIN or nxt['note'] > GUITAR_MIDI_MAX:
            if (i + 1) not in remove_indices:
                remove_indices.add(i + 1)
                removed_details.append({
                    'index': i + 1, 'note': nxt['note'],
                    'start': nxt['start'], 'end': nxt['end'],
                    'reason': 'out_of_guitar_range'
                })
            continue

        pos_a = positions_map[i]
        pos_b = positions_map[i + 1]

        if not pos_a or not pos_b:
            continue

        fret_dist, _, _ = min_fret_distance(pos_a, pos_b)

        # 프렛 거리 0이면 같은 포지션 → OK
        if fret_dist == 0:
            continue

        # 시간 갭 계산 (초)
        time_gap = (nxt['start'] - curr['end']) * hop_length / sr
        # 노트가 겹치면 시작 시간 차이 사용
        if time_gap <= 0:
            time_gap = (nxt['start'] - curr['start']) * hop_length / sr
        time_gap = max(time_gap, 0.001)  # epsilon

        required_speed = fret_dist / time_gap

        if required_speed <= max_fret_speed:
            continue

        # 물리적으로 불가능한 이동 감지!
        # 둘 중 하나를 제거 - 점수 낮은 쪽

        score_curr = _removal_score(curr, sr, hop_length)
        score_nxt = _removal_score(nxt, sr, hop_length)

        # 보호 조건 확인
        curr_duration = curr['end'] - curr['start']
        nxt_duration = nxt['end'] - nxt['start']

        curr_protected = (
            curr_duration >= protect_long_frames or
            curr.get('confidence', 0) >= min_confidence_protect
        )
        nxt_protected = (
            nxt_duration >= protect_long_frames or
            nxt.get('confidence', 0) >= min_confidence_protect
        )

        # 둘 다 보호되면 건너뜀
        if curr_protected and nxt_protected:
            continue

        if nxt_protected or (not curr_protected and score_curr < score_nxt):
            target_idx = i
            target = curr
        else:
            target_idx = i + 1
            target = nxt

        if target_idx not in remove_indices:
            remove_indices.add(target_idx)
            removed_details.append({
                'index': target_idx,
                'note': target['note'],
                'start': target['start'],
                'end': target['end'],
                'reason': 'fret_speed_exceeded',
                'required_speed': round(required_speed, 1),
                'max_allowed': max_fret_speed,
                'fret_distance': fret_dist,
                'time_gap_ms': round(time_gap * 1000, 1),
            })

    # 3. 필터링 적용
    filtered = [evt for i, evt in enumerate(events) if i not in remove_indices]

    report = {
        'original_count': len(events),
        'filtered_count': len(filtered),
        'removed_count': len(remove_indices),
        'removed_notes': removed_details,
        'max_fret_speed': max_fret_speed,
    }

    return (filtered, report)


def _empty_report(count):
    return {
        'original_count': count,
        'filtered_count': count,
        'removed_count': 0,
        'removed_notes': [],
        'max_fret_speed': 0,
    }
