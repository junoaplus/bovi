#!/usr/bin/env python3
"""
파인튜닝 모델 테스트 스크립트
새로 업데이트된 minjeongHuggingFace/koalpaca-bang_e9 모델을 테스트합니다.
"""

import sys
import os
import asyncio
import logging

# 프로젝트 루트 디렉터리를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.finetuning_service import FinetuningService

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_finetuning_model():
    """파인튜닝 모델 테스트"""
    
    print("🧪 파인튜닝 모델 테스트를 시작합니다...")
    print("=" * 60)
    
    try:
        # 1. 서비스 초기화
        print("1️⃣ 파인튜닝 서비스 초기화 중...")
        service = FinetuningService()
        
        # 2. 모델 정보 확인
        print("\\n2️⃣ 모델 정보 확인:")
        model_info = service.get_model_info()
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        if not model_info["model_loaded"]:
            print("❌ 모델이 로드되지 않았습니다. 테스트를 중단합니다.")
            return
        
        # 3. 테스트 질문들
        test_cases = [
            {
                "game_name": "방 탈출",
                "question": "게임 시작은 어떻게 하나요?"
            },
            {
                "game_name": "bang",
                "question": "보안관의 역할은 무엇인가요?"
            },
            {
                "game_name": "방 탈출",
                "question": "카드는 몇 장까지 가질 수 있나요?"
            }
        ]
        
        print("\\n3️⃣ 테스트 질문 답변:")
        print("-" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\\n테스트 {i}:")
            print(f"🎮 게임: {test_case['game_name']}")
            print(f"❓ 질문: {test_case['question']}")
            
            # 답변 생성
            answer = await service.answer_question(
                test_case["game_name"], 
                test_case["question"]
            )
            
            print(f"🤖 답변: {answer}")
            print("-" * 40)
        
        print("\\n✅ 테스트 완료!")
        
    except Exception as e:
        print(f"\\n❌ 테스트 중 오류 발생: {str(e)}")
        logger.error(f"테스트 오류: {str(e)}")

def main():
    """메인 함수"""
    print("🚀 파인튜닝 모델 테스트 시작")
    print(f"📍 모델: minjeongHuggingFace/koalpaca-bang_e9")
    print(f"🕐 시작 시간: {asyncio.get_event_loop().time()}")
    
    # 비동기 테스트 실행
    asyncio.run(test_finetuning_model())

if __name__ == "__main__":
    main()
