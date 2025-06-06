import json
import faiss
import numpy as np
import os
import re
import openai
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RAGService:
    """RAG 기반 게임 추천 및 룰 설명 서비스"""
    
    def __init__(self):
        logger.info("🔧 RAG 서비스를 초기화합니다...")
        
        # 임베딩 모델 로드
        self.embed_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
        logger.info("✅ 임베딩 모델 로드 완료")
        
        # OpenAI 설정
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_id = os.getenv("FINETUNING_MODEL_ID", "ft:gpt-3.5-turbo-0125:tset::BX2RnWfq")
        
        # 게임 추천용 데이터 로드
        self._load_recommendation_data()
        
        # 게임 룰 데이터 로드
        self._load_game_rules_data()
        
        logger.info("✅ RAG 서비스 초기화 완료")
    
    def _load_recommendation_data(self):
        """게임 추천용 데이터 로드"""
        try:
            # 게임 추천용 FAISS 인덱스
            index_path = "data/game_index.faiss"
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info("✅ 게임 추천 인덱스 로드 완료")
            else:
                logger.warning("⚠️ 게임 추천 인덱스 파일이 없습니다.")
                self.index = None
            
            # 게임 텍스트 데이터
            texts_path = "data/texts.json"
            if os.path.exists(texts_path):
                with open(texts_path, "r", encoding="utf-8") as f:
                    self.texts = json.load(f)
                logger.info("✅ 게임 텍스트 데이터 로드 완료")
            else:
                logger.warning("⚠️ 게임 텍스트 파일이 없습니다.")
                self.texts = []
            
            # 게임 이름 데이터
            names_path = "data/game_names.json"
            if os.path.exists(names_path):
                with open(names_path, "r", encoding="utf-8") as f:
                    self.game_names = json.load(f)
                logger.info("✅ 게임 이름 데이터 로드 완료")
            else:
                logger.warning("⚠️ 게임 이름 파일이 없습니다.")
                self.game_names = []
                
        except Exception as e:
            logger.error(f"❌ 게임 추천 데이터 로드 실패: {str(e)}")
            self.index = None
            self.texts = []
            self.game_names = []
    
    def _load_game_rules_data(self):
        """게임 룰 데이터 로드"""
        try:
            # 게임 전체 룰 데이터
            game_data_path = "data/game.json"
            if os.path.exists(game_data_path):
                with open(game_data_path, "r", encoding="utf-8") as f:
                    self.game_data = json.load(f)
                logger.info("✅ 게임 룰 데이터 로드 완료")
            else:
                logger.warning("⚠️ 게임 룰 파일이 없습니다.")
                self.game_data = []
            
            # 게임별 벡터 인덱스 경로
            self.game_vector_base_path = "data/game_data/game_data"
            
        except Exception as e:
            logger.error(f"❌ 게임 룰 데이터 로드 실패: {str(e)}")
            self.game_data = []
    
    async def recommend_games(self, query: str, top_k: int = 3):
        """게임 추천"""
        try:
            if not self.index or not self.texts or not self.game_names:
                return "게임 추천 데이터가 로드되지 않았습니다."
            
            # 쿼리에서 개수 추출
            number_match = re.search(r'(\d+)\s*개', query)
            if number_match:
                top_k = int(number_match.group(1))
            
            # 1. 쿼리 임베딩 → 유사도 검색
            query_embedding = self.embed_model.encode([query], normalize_embeddings=True)
            D, I = self.index.search(np.array(query_embedding), k=top_k)
            
            # 2. 검색된 게임 설명과 이름들 추출
            retrieved = []
            selected_names = []
            for i in I[0]:
                if i < len(self.game_names) and i < len(self.texts):
                    name = self.game_names[i]
                    desc = self.texts[i]
                    retrieved.append(f"[{name}]\n{desc}")
                    selected_names.append(f"- {name}")
            
            context = "\n\n".join(retrieved)
            name_list_str = "\n".join(selected_names)
            
            # 3. 프롬프트 생성
            prompt = f"""아래는 보드게임 설명입니다. 각 게임은 "[게임명]\\n설명" 형식으로 되어 있습니다.

[게임 설명]
{context}

⚠️ 반드시 아래의 게임 이름 목록 중에서만 선택할 수 있습니다. 이 목록에 없는 게임 이름을 절대 생성하지 마세요.
만약 목록에 없는 게임명을 출력하면 실패로 간주됩니다.

[허용된 게임 이름 목록]
{name_list_str}

[사용자 질문]
{query}

📌 출력 지침:
- 반드시 위 목록에 있는 게임 중에서만 {top_k}개를 골라 추천하세요.
- 출력 형식은 반드시 아래 형식처럼 작성하세요:

게임명1: 추천 이유
게임명2: 추천 이유
게임명3: 추천 이유

- 각 줄은 '게임명: 추천 이유' 형식으로만 작성하고, 줄바꿈 이외에 아무 포맷도 쓰지 마세요.
- 추천이 모두 끝나면 마지막 줄에 반드시 다음과 같이 써주세요:
추천 완료!

그 이후에는 아무 것도 쓰지 마세요.
"""
            
            # 4. GPT 호출
            response = openai.ChatCompletion.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "너는 보드게임 추천 도우미야. 사용자의 질문에 따라 관련 게임을 추천하고 이유도 알려줘."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=768
            )
            
            raw_output = response["choices"][0]["message"]["content"]
            
            # 5. 출력 후처리
            if "추천 완료!" in raw_output:
                raw_output = raw_output.split("추천 완료!")[0]
            
            return raw_output.strip()
            
        except Exception as e:
            logger.error(f"❌ 게임 추천 실패: {str(e)}")
            return f"게임 추천 중 오류가 발생했습니다: {str(e)}"
    
    async def answer_rule_question(self, game_name: str, question: str):
        """룰 질문 답변"""
        try:
            # 게임별 벡터 인덱스 로드
            game_index_path = os.path.join(self.game_vector_base_path, f"{game_name}.faiss")
            game_chunks_path = os.path.join(self.game_vector_base_path, f"{game_name}.json")
            
            if not os.path.exists(game_index_path) or not os.path.exists(game_chunks_path):
                return f"'{game_name}' 게임의 데이터를 찾을 수 없습니다."
            
            # 벡터 인덱스 및 청크 텍스트 로딩
            index = faiss.read_index(game_index_path)
            with open(game_chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            # RAG 검색
            q_vec = self.embed_model.encode([question], normalize_embeddings=True)
            D, I = index.search(np.array(q_vec), k=3)
            retrieved_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
            
            context = "\n\n".join(retrieved_chunks)
            
            # 프롬프트 생성
            rag_prompt = f"""
아래는 '{game_name}' 보드게임의 룰 설명 일부입니다:

{context}

이 룰을 바탕으로 다음 질문에 정확하고 구체적으로 답변해줘:

질문: {question}
"""
            
            # GPT 호출
            response = openai.ChatCompletion.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": (
                        "너는 보드게임 룰 전문 AI야. 반드시 아래 규칙을 따라야 해:\n"
                        "- 사용자의 질문에 대해 아래 룰 설명(context)에 있는 내용만 기반해서 답변해.\n"
                        "- 룰 설명에 없는 정보는 절대로 지어내거나 상상하지 마.\n"
                        "- 사람 이름, 장소, 시간, 인원수 등을 추측하거나 새로 만들어내지 마.\n"
                        "- 답할 수 없는 질문이면 '해당 정보는 룰에 명시되어 있지 않습니다.' 라고 말해."
                    )},
                    {"role": "user", "content": rag_prompt}
                ],
                temperature=0.7,
                max_tokens=768
            )
            
            answer = response["choices"][0]["message"]["content"]
            return answer.strip()
            
        except Exception as e:
            logger.error(f"❌ 룰 질문 답변 실패: {str(e)}")
            return f"룰 질문 답변 중 오류가 발생했습니다: {str(e)}"
    
    async def get_rule_summary(self, game_name: str, chat_type: str = "gpt"):
        """게임 룰 요약"""
        try:
            # 게임 정보 찾기
            game_info = None
            for game in self.game_data:
                if game.get("game_name") == game_name:
                    game_info = game
                    break
            
            if not game_info:
                return f"'{game_name}' 게임의 정보를 찾을 수 없습니다."
            
            game_rule_text = game_info.get('text', '')
            
            # 시스템 프롬프트 설정
            system_prompt = (
                "너는 보드게임 룰 전문 AI야. 반드시 아래 규칙을 따라야 해:\n"
                "- 사용자가 선택한 보드게임의 룰 전체를 보고, 그 게임의 룰을 알기 쉽게 설명해줘.\n"
                "- 핵심 개념, 목표, 진행 방식, 승리 조건을 요약해줘.\n"
                "- 설명은 간결하고 구조적으로 작성해."
            )
            
            # 프롬프트 생성
            prompt = f"게임 이름: {game_name}\n\n룰 전체:\n{game_rule_text}\n\n이 게임의 룰을 설명해주세요."
            
            # GPT 호출
            response = openai.ChatCompletion.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=768
            )
            
            summary = response["choices"][0]["message"]["content"]
            return summary.strip()
            
        except Exception as e:
            logger.error(f"❌ 룰 요약 실패: {str(e)}")
            return f"룰 요약 중 오류가 발생했습니다: {str(e)}"
    
    def get_available_games(self):
        """사용 가능한 게임 목록 반환"""
        if self.game_names:
            return self.game_names
        elif self.game_data:
            return [game.get("game_name", "") for game in self.game_data if game.get("game_name")]
        else:
            # 기본 게임 목록
            return [
                "카탄", "스플렌더", "아줄", "윙스팬", "뱅", 
                "킹 오브 도쿄", "7 원더스", "도미니언", "스몰 월드", "티켓 투 라이드"
            ]
