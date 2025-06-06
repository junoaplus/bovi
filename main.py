from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import logging
from typing import List, Optional

# 서비스 import
from services.embedding_service import EmbeddingService
from services.finetuning_service import FinetuningService
from services.rag_service import RAGService

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
class GameRecommendationRequest(BaseModel):
    query: str
    top_k: int = 3

class RuleQuestionRequest(BaseModel):
    game_name: str
    question: str
    chat_type: str = "gpt"

class GameRuleSummaryRequest(BaseModel):
    game_name: str
    chat_type: str = "gpt"

class APIResponse(BaseModel):
    status: str
    data: Optional[dict] = None
    message: Optional[str] = None

# 전역 변수로 서비스 인스턴스 저장
services_initialized = False
embedding_service = None
finetuning_service = None
rag_service = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 AI 모델들을 로드"""
    global services_initialized
    logger.info("🚀 AI 백엔드 서버를 시작합니다...")
    
    try:
        # 서비스 초기화 (실제 모델 로드)
        global embedding_service, finetuning_service, rag_service
        
        # RAG 서비스는 필수 (게임 추천 및 룰 설명)
        rag_service = RAGService()
        
        # 파인튜닝 서비스는 선택사항 (모델 파일이 있을 때만)
        try:
            finetuning_service = FinetuningService()
        except Exception as e:
            logger.warning(f"⚠️ 파인튜닝 서비스 로드 실패 (계속 진행): {str(e)}")
            finetuning_service = None
        
        # 임베딩 서비스는 현재 RAG에 포함되어 있어 별도 로드하지 않음
        # embedding_service = EmbeddingService()
        
        services_initialized = True
        logger.info("✅ 모든 AI 서비스가 성공적으로 초기화되었습니다!")
        
    except Exception as e:
        logger.error(f"❌ 서비스 초기화 실패: {str(e)}")
        services_initialized = False

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy" if services_initialized else "initializing",
        "services_loaded": services_initialized,
        "message": "보드게임 AI 백엔드가 정상 작동 중입니다!"
    }

@app.post("/recommend", response_model=APIResponse)
async def recommend_games(request: GameRecommendationRequest):
    """게임 추천 API"""
    try:
        if not services_initialized:
            raise HTTPException(status_code=503, detail="서비스가 아직 초기화되지 않았습니다.")
        
        logger.info(f"게임 추천 요청: {request.query}")
        
        # RAG 서비스 호출
        result = await rag_service.recommend_games(request.query, request.top_k)
        
        return APIResponse(
            status="success",
            data={"recommendation": result},
            message="게임 추천이 완료되었습니다."
        )
        
    except Exception as e:
        logger.error(f"게임 추천 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"게임 추천 중 오류가 발생했습니다: {str(e)}")

@app.post("/explain-rules", response_model=APIResponse)
async def explain_rules(request: RuleQuestionRequest):
    """룰 설명 API"""
    try:
        if not services_initialized:
            raise HTTPException(status_code=503, detail="서비스가 아직 초기화되지 않았습니다.")
        
        logger.info(f"룰 질문: {request.game_name} - {request.question}")
        
        # 서비스 호출
        if request.chat_type == "finetuning" and finetuning_service:
            result = await finetuning_service.answer_question(request.game_name, request.question)
        else:
            result = await rag_service.answer_rule_question(request.game_name, request.question)
        
        return APIResponse(
            status="success",
            data={"answer": result},
            message="룰 설명이 완료되었습니다."
        )
        
    except Exception as e:
        logger.error(f"룰 설명 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"룰 설명 중 오류가 발생했습니다: {str(e)}")

@app.post("/rule-summary", response_model=APIResponse)
async def get_rule_summary(request: GameRuleSummaryRequest):
    """게임 룰 요약 API"""
    try:
        if not services_initialized:
            raise HTTPException(status_code=503, detail="서비스가 아직 초기화되지 않았습니다.")
        
        logger.info(f"룰 요약 요청: {request.game_name}")
        
        # RAG 서비스 호출
        result = await rag_service.get_rule_summary(request.game_name, request.chat_type)
        
        return APIResponse(
            status="success",
            data={"summary": result},
            message="룰 요약이 완료되었습니다."
        )
        
    except Exception as e:
        logger.error(f"룰 요약 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"룰 요약 중 오류가 발생했습니다: {str(e)}")

@app.get("/games")
async def get_available_games():
    """사용 가능한 게임 목록 API"""
    try:
        # 게임 목록 로드
        games = rag_service.get_available_games()
        
        return APIResponse(
            status="success",
            data={"games": games},
            message=f"총 {len(games)}개의 게임을 지원합니다."
        )
        
    except Exception as e:
        logger.error(f"게임 목록 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"게임 목록 조회 중 오류가 발생했습니다: {str(e)}")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "보드게임 AI 백엔드 서버",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend",
            "explain_rules": "/explain-rules",
            "rule_summary": "/rule-summary",
            "games": "/games"
        }
    }

if __name__ == "__main__":
    # 개발용 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8888,
        reload=True,
        log_level="info"
    )
