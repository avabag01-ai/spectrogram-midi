"""
새로 추가된 기능 테스트 스크립트
- Auto Parameter Matcher
- Reverse Analyzer
"""

import sys
import os

# 임포트 테스트
try:
    print("=" * 60)
    print("📦 모듈 임포트 테스트")
    print("=" * 60)

    from aegis_engine_core.auto_matcher import auto_match_parameters
    print("✅ auto_matcher 모듈 임포트 성공")

    from aegis_engine_core.reverse_analyzer import reverse_analysis
    print("✅ reverse_analyzer 모듈 임포트 성공")

    from aegis_engine import AegisEngine
    print("✅ AegisEngine 임포트 성공")

    from aegis_engine_core.synthesizer import get_synthesizer, synthesize_midi
    print("✅ synthesizer 모듈 임포트 성공")

    print("\n" + "=" * 60)
    print("✅ 모든 모듈 임포트 성공!")
    print("=" * 60)

    # FluidSynth 사용 가능 여부 확인
    print("\n🔍 FluidSynth 상태 확인...")
    synth = get_synthesizer()
    if synth.is_available():
        print("✅ FluidSynth 사용 가능")
        print(f"   경로: {synth.fluidsynth_path}")
        print(f"   SoundFont: {synth.soundfont}")
    else:
        print("⚠️  FluidSynth를 찾을 수 없습니다")
        print("   설치: brew install fluid-synth")

    print("\n" + "=" * 60)
    print("🎉 테스트 완료!")
    print("=" * 60)
    print("\n📌 사용법:")
    print("   1. streamlit run aegis_tuner_pro.py 실행")
    print("   2. 음원 파일 선택")
    print("   3. '🤖 Auto Match' 버튼 클릭 → 자동 파라미터 최적화")
    print("   4. '🔬 역변환 분석 실행' 버튼 클릭 → MIDI 정확도 분석")

except ImportError as e:
    print(f"❌ 임포트 실패: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
