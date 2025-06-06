# 테스트용 간단한 main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="보드게임 AI 백엔드",
    description="Runpod에서 실행되는 보드게임 추천 및 룰 설명 AI 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response 모델들
class APIResponse(BaseModel):
    status: str
    data: Optional[dict] = None
    message: Optional[str] = None

# 전역 변수로 서비스 상태 저장
services_initialized = False
rag_service = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 AI 모델들을 로드"""
    global services_initialized, rag_service
    logger.info("🚀 AI 백엔드 서버를 시작합니다...")
    
    try:
        # 서비스 초기화 시도
        from services.rag_service import RAGService
        
        logger.info("📚 RAG 서비스를 초기화합니다...")
        rag_service = RAGService()
        
        # 초기화 성공 여부 확인
        if hasattr(rag_service, 'initialized') and rag_service.initialized:
            services_initialized = True
            logger.info("✅ RAG 서비스 초기화 성공!")
        else:
            logger.warning("⚠️ RAG 서비스 초기화가 완전하지 않음")
            services_initialized = False
        
    except Exception as e:
        logger.error(f"❌ 서비스 초기화 실패: {str(e)}")
        services_initialized = False
        rag_service = None

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy" if services_initialized else "initializing",
        "services_loaded": services_initialized,
        "rag_service_available": rag_service is not None,
        "message": "보드게임 AI 백엔드가 정상 작동 중입니다!"
    }

@app.get("/games")
async def get_available_games():
    """사용 가능한 게임 목록 API"""
    try:
        logger.info("🎮 게임 목록 요청 받음")
        
        # 기본 게임 목록 (서비스 초기화 실패 시에도 사용)
        default_games = [
            "카탄", "스플렌더", "아줄", "윙스팬", "뱅", 
            "킹 오브 도쿄", "7 원더스", "도미니언", "스몰 월드", "티켓 투 라이드"
        ]
        
        # RAG 서비스가 초기화되어 있으면 해당 메서드 호출 시도
        if services_initialized and rag_service is not None:
            try:
                if hasattr(rag_service, 'get_available_games'):
                    games = rag_service.get_available_games()
                    logger.info(f"✅ RAG 서비스에서 {len(games)}개 게임 로드")
                    
                    if games:  # 게임 목록이 비어있지 않으면 사용
                        return APIResponse(
                            status="success",
                            data={"games": games},
                            message=f"총 {len(games)}개의 게임을 지원합니다."
                        )
                else:
                    logger.warning("⚠️ RAG 서비스에 get_available_games 메서드가 없음")
            except Exception as e:
                logger.error(f"❌ RAG 서비스 호출 실패: {str(e)}")
        
        # 기본 게임 목록 반환
        logger.info("📋 기본 게임 목록 사용")
        return APIResponse(
            status="success",
            data={"games": default_games},
            message=f"총 {len(default_games)}개의 기본 게임을 지원합니다."
        )
        
    except Exception as e:
        logger.error(f"❌ 게임 목록 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"게임 목록 조회 중 오류가 발생했습니다: {str(e)}")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "보드게임 AI 백엔드 서버",
        "version": "1.0.0",
        "status": "running",
        "services_initialized": services_initialized,
        "endpoints": {
            "health": "/health",
            "games": "/games"
        }
    }

if __name__ == "__main__":
    import os
    
    # 환경변수 설정
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 보드게임 AI 백엔드 서버를 시작합니다...")
    print(f"📡 서버 주소: http://{host}:{port}")
    print(f"📚 API 문서: http://{host}:{port}/docs")
    print(f"🔍 헬스체크: http://{host}:{port}/health")
    print("⏰ 모델 로딩에 30초~2분 정도 소요됩니다...")
    print("="*50)
    
    # 서버 실행
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # 프로덕션에서는 reload=False
        log_level="info",
        access_log=True
    )
