import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

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
        """파인튜닝된 모델 로드 (base 모델 + PEFT 어댑터)"""
        try:
            finetuned_model_name = os.getenv("FINETUNING_MODEL_ID")
            logger.info(f"📥 파인튜닝된 어댑터 로드 중: {finetuned_model_name}")
            
            # 1. base 모델 명 지정 (환경변수 or 코드 내 기본값)
            base_model_name = "beomi/KoAlpaca-Polyglot-5.8B"
            logger.info(f"📥 base 모델 로드 중: {base_model_name}")
            
            # 2. base 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 3. adapter(PEFT) 모델 로드하여 base 모델에 씌우기
            self.model = PeftModel.from_pretrained(
                base_model,
                finetuned_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 4. tokenizer 로드 (기본적으로 어댑터가 아니라 base 모델 tokenizer 사용)
            logger.info("📥 Tokenizer 로드 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_fast=False,
                trust_remote_code=True
            )
            
            logger.info("✅ 파인튜닝 모델(어댑터) 로드 완료")
                
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {str(e)}")
            logger.info("🔄 기본 모델로 대체 시도...")
            
            try:
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
                self.model = None
                self.tokenizer = None
    
    async def answer_question(self, game_name: str, question: str):
        """파인튜닝된 모델로 질문 답변"""
        try:
            if not self.model or not self.tokenizer:
                return "파인튜닝 모델이 로드되지 않았습니다."
            
            prompt = f"이 질문은 '{game_name}'이라는 보드게임에 대한 것이다.\n### 질문: {question}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            inputs.pop("token_type_ids", None)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            result = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = result.replace(prompt, "").strip()
            
            if not answer:
                answer = f"'{game_name}' 게임에 대한 '{question}' 질문에 대한 답변을 생성할 수 없습니다."
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ 파인튜닝 모델 답변 생성 실패: {str(e)}")
            return f"파인튜닝 모델 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def get_model_info(self):
        return {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "device": self.device,
            "finetuned_model": os.getenv("FINETUNING_MODEL_ID"),
            "base_model": os.getenv("BASE_MODEL_ID", "beomi/KoAlpaca-Polyglot-5.8B")
        }
