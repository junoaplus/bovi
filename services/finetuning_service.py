import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class FinetuningService:
    """파인튜닝된 모델을 사용한 룰 설명 서비스"""
    
    def __init__(self):
        logger.info("🔧 파인튜닝 서비스를 초기화합니다...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"💻 디바이스: {self.device}")
        
        # 모델 로드
        self._load_model()
        
        logger.info("✅ 파인튜닝 서비스 초기화 완료")
    
    def _load_model(self):
        """파인튜닝된 모델 로드"""
        try:
            # 허깅페이스에서 직접 파인튜닝된 모델 로드
            finetuned_model_name = "minjeongHuggingFace/koalpaca-bang_e9"
            logger.info(f"📥 파인튜닝된 모델 로드 중: {finetuned_model_name}")
            
            # 1. 파인튜닝된 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                finetuned_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 2. Tokenizer 로드 (같은 모델에서)
            logger.info("📥 Tokenizer 로드 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                finetuned_model_name, 
                use_fast=False,
                trust_remote_code=True
            )
            
            logger.info("✅ 파인튜닝 모델 로드 완료")
                
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {str(e)}")
            logger.info("🔄 기본 모델로 대체 시도...")
            
            try:
                # 기본 KoAlpaca 모델로 대체
                base_model_name = "beomi/KoAlpaca-Polyglot-5.8B"
                logger.info(f"📥 기본 모델 로드: {base_model_name}")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name, 
                    use_fast=False,
                    trust_remote_code=True
                )
                
                logger.info("✅ 기본 모델 로드 완료")
                
            except Exception as backup_e:
                logger.error(f"❌ 기본 모델 로드도 실패: {str(backup_e)}")
                # 모델 로드 실패 시 None으로 설정
                self.model = None
                self.tokenizer = None
    
    async def answer_question(self, game_name: str, question: str):
        """파인튜닝된 모델로 질문 답변"""
        try:
            if not self.model or not self.tokenizer:
                return "파인튜닝 모델이 로드되지 않았습니다."
            
            # 프롬프트 생성
            prompt = f"이 질문은 '{game_name}'이라는 보드게임에 대한 것이다.\n### 질문: {question}"
            
            # 토큰화
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # token_type_ids가 있으면 제거 (일부 모델에서 에러 방지)
            inputs.pop("token_type_ids", None)
            
            # 추론
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,    # 샘플링 OFF
                    temperature=0.0,    # 완전 낮춤
                    top_p=1.0,          # 전부 허용
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 결과 디코딩
            result = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # 프롬프트 부분 제거하고 답변만 추출
            answer = result.replace(prompt, "").strip()
            
            # 답변이 비어있으면 기본 메시지
            if not answer:
                answer = f"'{game_name}' 게임에 대한 '{question}' 질문에 대한 답변을 생성할 수 없습니다."
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ 파인튜닝 모델 답변 생성 실패: {str(e)}")
            return f"파인튜닝 모델 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "device": self.device,
            "finetuned_model": "minjeongHuggingFace/koalpaca-bang_e9",
            "fallback_model": "beomi/KoAlpaca-Polyglot-5.8B"
        }
