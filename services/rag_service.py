import json
import faiss
import numpy as np
import os
import re
import logging
from sentence_transformers import SentenceTransformer

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# 세션 기반 클래스 메모리 정의 (LangChain용)
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_messages(self, messages):
        self.messages.extend(messages)

    def clear(self):
        self.messages = []

    def __repr__(self):
        return str(self.messages)

# 세션 저장소 (LangChain용)
store = {}
def get_session_history_for_rag(session_id: str) -> InMemoryHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


class RAGService:
    """RAG 기반 게임 추천 및 룰 설명 서비스"""
    
    def __init__(self):
        logger.info("🔧 RAG 서비스를 초기화합니다...")
        
        # 임베딩 모델 로드
        self.embed_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
        logger.info("✅ 임베딩 모델 로드 완료")
        
        # OpenAI 설정 (LangChain ChatOpenAI 사용)
        self.openai_api_key = ""# 환경 변수 사용 권장
        self.model_id = "gpt-3.5-turbo" # 파인튜닝 모델 ID
        self.llm = ChatOpenAI(model_name=self.model_id, temperature=0.7, openai_api_key=self.openai_api_key)
        
        # 게임 추천용 데이터 로드
        self._load_recommendation_data()
        
        # 게임 룰 데이터 로드
        self._load_game_rules_data()
        
        # LangChain 체인 설정
        self._setup_langchain_chains()
        
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
                logger.warning("⚠️ 게임 추천 인덱스 파일이 없습니다. 'game_index.faiss' 경로를 확인하세요.")
                self.index = None
            
            # 게임 텍스트 데이터
            texts_path = "data/texts.json"
            if os.path.exists(texts_path):
                with open(texts_path, "r", encoding="utf-8") as f:
                    self.texts = json.load(f)
                logger.info("✅ 게임 텍스트 데이터 로드 완료")
            else:
                logger.warning("⚠️ 게임 텍스트 파일이 없습니다. 'texts.json' 경로를 확인하세요.")
                self.texts = []
            
            # 게임 이름 데이터
            names_path = "data/game_names.json"
            if os.path.exists(names_path):
                with open(names_path, "r", encoding="utf-8") as f:
                    self.game_names = json.load(f)
                logger.info("✅ 게임 이름 데이터 로드 완료")
            else:
                logger.warning("⚠️ 게임 이름 파일이 없습니다. 'game_names.json' 경로를 확인하세요.")
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
            game_data_path = "data/game.json" # 모든 게임의 상세 룰이 담긴 파일
            if os.path.exists(game_data_path):
                with open(game_data_path, "r", encoding="utf-8") as f:
                    self.game_data = json.load(f)
                logger.info("✅ 게임 룰 데이터 로드 완료")
            else:
                logger.warning("⚠️ 게임 룰 파일이 없습니다. 'game.json' 경로를 확인하세요.")
                self.game_data = []
            
            # 게임별 벡터 인덱스 경로 (개별 게임 룰 청크를 위한 폴더)
            self.game_vector_base_path = "data/game_data/game_data"
            
        except Exception as e:
            logger.error(f"❌ 게임 룰 데이터 로드 실패: {str(e)}")
            self.game_data = []

    def _setup_langchain_chains(self):
        """LangChain 체인 및 프롬프트 설정"""
        # 게임 추천 프롬프트 (search_similar_context의 결과를 {context}로 받음)
        recommendation_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "너는 보드게임 추천 도우미야. 다음은 추천 가능한 게임 설명들이야:\n\n{context}\n\n"
                "반드시 이 게임 목록 안에서만 추천해. 새로운 게임을 지어내지 마.\n"
                "질문에 맞는 게임 {top_k}개를 골라 아래 형식으로 답해:\n"
                "게임명1: 이유\n게임명2: 이유\n게임명3: 이유\n"
                "각 줄은 '게임명: 추천 이유' 형식으로만 작성하고, 줄바꿈 이외에 아무 포맷도 쓰지 마세요.\n"
                "추천이 모두 끝나면 마지막 줄에 반드시 '추천 완료!' 라고 써주세요. 그 이후에는 아무 것도 쓰지 마세요."
            ),
            MessagesPlaceholder(variable_name="history"), # 세션 히스토리
            ("human", "{query}")
        ])

        # 게임 추천 체인 (RunnableWithMessageHistory로 히스토리 관리)
        self.recommendation_chain = RunnableWithMessageHistory(
            recommendation_prompt | self.llm,
            get_session_history=get_session_history_for_rag,
            input_messages_key="query",  # 사용자의 실제 입력 쿼리
            history_messages_key="history" # 프롬프트의 히스토리 placeholder
        )

        # 룰 질문 답변 프롬프트 (룰 청크 검색 결과를 {context}로 받음)
        rule_question_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "너는 보드게임 룰 전문 AI야. 반드시 아래 규칙을 따라야 해:\n"
                "- 사용자의 질문에 대해 아래 룰 설명(context)에 있는 내용만 기반해서 답변해.\n"
                "- 룰 설명에 없는 정보는 절대로 지어내거나 상상하지 마.\n"
                "- 사람 이름, 장소, 시간, 인원수 등을 추측하거나 새로 만들어내지 마.\n"
                "- 답할 수 없는 질문이면 '해당 정보는 룰에 명시되어 있지 않습니다.' 라고 말해."
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "아래는 '{game_name}' 보드게임의 룰 설명 일부입니다:\n\n{context}\n\n이 룰을 바탕으로 다음 질문에 정확하고 구체적으로 답변해줘:\n\n질문: {question}")
        ])

        # 룰 질문 답변 체인
        self.rule_question_chain = RunnableWithMessageHistory(
            rule_question_prompt | self.llm,
            get_session_history=get_session_history_for_rag,
            input_messages_key="question",
            history_messages_key="history"
        )

        # 룰 요약 프롬프트 (전체 룰 텍스트를 {game_rule_text}로 받음)
        rule_summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "너는 보드게임 룰 전문 AI야. 반드시 아래 규칙을 따라야 해:\n"
                "- 사용자가 선택한 보드게임의 룰 전체를 보고, 그 게임의 룰을 알기 쉽게 설명해줘.\n"
                "- 핵심 개념, 목표, 진행 방식, 승리 조건을 요약해줘.\n"
                "- 설명은 간결하고 구조적으로 작성해."
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "게임 이름: {game_name}\n\n룰 전체:\n{game_rule_text}\n\n이 게임의 룰을 설명해주세요.")
        ])

        # 룰 요약 체인
        self.rule_summary_chain = RunnableWithMessageHistory(
            rule_summary_prompt | self.llm,
            get_session_history=get_session_history_for_rag,
            input_messages_key="game_name", # 게임 이름이 주 입력값이 됨
            history_messages_key="history"
        )

    def _search_similar_context(self, query, top_k=3):
        """
        첫 번째 코드의 search_similar_context 함수와 동일한 RAG 검색 로직.
        쿼리를 임베딩하여 FAISS 인덱스에서 유사한 게임 설명을 찾습니다.
        """
        if not self.index or not self.texts or not self.game_names:
            logger.warning("RAG 검색을 위한 인덱스나 텍스트 데이터가 로드되지 않았습니다.")
            return ""

        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(query_vec), top_k)

        context_blocks = []
        for i in I[0]:
            if 0 <= i < len(self.game_names) and 0 <= i < len(self.texts):
                context_blocks.append(f"[{self.game_names[i]}]\n{self.texts[i]}")
            else:
                logger.warning(f"인덱스 {i}에 해당하는 게임 이름 또는 텍스트를 찾을 수 없습니다.")
        return "\n\n".join(context_blocks)
    
    async def recommend_games(self, query: str, session_id: str = "default_session", top_k: int = 3):
        """게임 추천 (RAG 검색 후 LangChain으로 LLM 호출)"""
        try:
            # 쿼리에서 추천 개수 추출
            number_match = re.search(r'(\d+)\s*개', query)
            if number_match:
                top_k = int(number_match.group(1))

            # 1. RAG 검색: query를 기반으로 유사한 게임 설명을 가져옴 (첫 번째 코드의 핵심 로직)
            context = self._search_similar_context(query, top_k=top_k)
            
            if not context:
                return "추천할 게임 데이터를 찾을 수 없습니다. 인덱스나 데이터 로드를 확인해주세요."

            # 2. LangChain 체인 호출: 검색된 context와 사용자 쿼리를 LLM에 전달
            response = await self.recommendation_chain.ainvoke(
                {"query": query, "context": context, "top_k": top_k},
                config={"configurable": {"session_id": session_id}}
            )
            
            raw_output = response.content
            
            # 3. 출력 후처리
            if "추천 완료!" in raw_output:
                raw_output = raw_output.split("추천 완료!")[0]
            
            return raw_output.strip()
            
        except Exception as e:
            logger.error(f"❌ 게임 추천 실패: {str(e)}")
            return f"게임 추천 중 오류가 발생했습니다: {str(e)}"
    
    async def answer_rule_question(self, game_name: str, question: str, session_id: str = "default_session"):
        """룰 질문 답변 (룰 청크 검색 후 LangChain으로 LLM 호출)"""
        try:
            # 게임별 벡터 인덱스 로드
            game_index_path = os.path.join(self.game_vector_base_path, f"{game_name}.faiss")
            game_chunks_path = os.path.join(self.game_vector_base_path, f"{game_name}.json")
            
            if not os.path.exists(game_index_path) or not os.path.exists(game_chunks_path):
                return f"'{game_name}' 게임의 룰 데이터를 찾을 수 없습니다. 해당 게임의 데이터가 올바른 경로에 있는지 확인해주세요."
            
            # 벡터 인덱스 및 청크 텍스트 로딩
            index = faiss.read_index(game_index_path)
            with open(game_chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            # RAG 검색: 룰 질문에 대한 유사 청크 검색
            q_vec = self.embed_model.encode([question], normalize_embeddings=True)
            D, I = index.search(np.array(q_vec), k=3)
            retrieved_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
            
            context = "\n\n".join(retrieved_chunks)
            
            if not context:
                return f"'{game_name}' 게임 룰에서 질문에 대한 관련 정보를 찾을 수 없습니다."

            # LangChain 체인 호출
            response = await self.rule_question_chain.ainvoke(
                {"game_name": game_name, "question": question, "context": context},
                config={"configurable": {"session_id": session_id}}
            )
            
            answer = response.content
            return answer.strip()
            
        except Exception as e:
            logger.error(f"❌ 룰 질문 답변 실패: {str(e)}")
            return f"룰 질문 답변 중 오류가 발생했습니다: {str(e)}"
    
    async def get_rule_summary(self, game_name: str, session_id: str = "default_session"):
        """게임 룰 요약 (전체 룰 텍스트를 LangChain으로 LLM 호출)"""
        try:
            # 게임 정보 찾기
            game_info = None
            for game in self.game_data:
                if game.get("game_name") == game_name:
                    game_info = game
                    break
            
            if not game_info:
                return f"'{game_name}' 게임의 전체 룰 정보를 찾을 수 없습니다. 'game.json' 파일을 확인해주세요."
            
            game_rule_text = game_info.get('text', '')
            
            if not game_rule_text:
                return f"'{game_name}' 게임의 룰 내용이 비어 있습니다."

            # LangChain 체인 호출
            response = await self.rule_summary_chain.ainvoke(
                {"game_name": game_name, "game_rule_text": game_rule_text},
                config={"configurable": {"session_id": session_id}}
            )
            
            summary = response.content
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