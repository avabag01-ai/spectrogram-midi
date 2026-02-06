"""
Aegis Engine Core v2.0 - Financial Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
주식 기술적 분석을 활용한 차세대 MIDI 변환 엔진

"로직 프로가 못 잡는 걸 주식으로 잡는다"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from .financial_analysis import FinancialPitchAnalyzer
from .midi_logic_financial import (
    get_midi_events_financial,
    detect_articulations_financial,
    adaptive_confidence_threshold
)

__all__ = [
    'FinancialPitchAnalyzer',
    'get_midi_events_financial',
    'detect_articulations_financial',
    'adaptive_confidence_threshold'
]

__version__ = '2.0.0-financial'
