"""
🎯 챗봇 서비스 - 구현 파일

이 파일은 챗봇의 핵심 AI 로직을 담당합니다.
아래 아키텍처를 참고하여 직접 설계하고 구현하세요.

📐 시스템 아키텍처:

┌─────────────────────────────────────────────────────────┐
│ 1. 초기화 단계 (ChatbotService.__init__)                  │
├─────────────────────────────────────────────────────────┤
│  - OpenAI Client 생성                                    │
│  - ChromaDB 연결 (벡터 데이터베이스)                       │
│  - LangChain Memory 초기화 (대화 기록 관리)               │
│  - Config 파일 로드                                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 2. RAG 파이프라인 (generate_response 내부)               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  사용자 질문 "학식 추천해줘"                              │
│       ↓                                                  │
│  [_create_embedding()]                                   │
│       ↓                                                  │
│  질문 벡터: [0.12, -0.34, ..., 0.78]  (3072차원)        │
│       ↓                                                  │
│  [_search_similar()]  ← ChromaDB 검색                    │
│       ↓                                                  │
│  검색 결과: "학식은 곤자가가 맛있어" (유사도: 0.87)        │
│       ↓                                                  │
│  [_build_prompt()]                                       │
│       ↓                                                  │
│  최종 프롬프트 = 시스템 설정 + RAG 컨텍스트 + 질문        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 3. LLM 응답 생성                                         │
├─────────────────────────────────────────────────────────┤
│  OpenAI GPT-4 API 호출                                   │
│       ↓                                                  │
│  "학식은 곤자가에서 먹는 게 제일 좋아! 돈까스가 인기야"    │
│       ↓                                                  │
│  [선택: 이미지 검색]                                      │
│       ↓                                                  │
│  응답 반환: {reply: "...", image: "..."}                 │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 메모리 저장 (LangChain Memory)                        │
├─────────────────────────────────────────────────────────┤
│  대화 기록에 질문-응답 저장                               │
│  다음 대화에서 컨텍스트로 활용                            │
└─────────────────────────────────────────────────────────┘


💡 핵심 구현 과제:

1. **Embedding 생성**
   - OpenAI API를 사용하여 텍스트를 벡터로 변환
   - 모델: text-embedding-3-large (3072차원)

2. **RAG 검색 알고리즘** ⭐ 가장 중요!
   - ChromaDB에서 유사 벡터 검색
   - 유사도 계산: similarity = 1 / (1 + distance)
   - threshold 이상인 문서만 선택

3. **LLM 프롬프트 설계**
   - 시스템 프롬프트 (캐릭터 설정)
   - RAG 컨텍스트 통합
   - 대화 기록 포함

4. **대화 메모리 관리**
   - LangChain의 ConversationSummaryBufferMemory 사용
   - 대화가 길어지면 자동으로 요약


📚 참고 문서:
- ARCHITECTURE.md: 시스템 아키텍처 상세 설명
- IMPLEMENTATION_GUIDE.md: 단계별 구현 가이드
- README.md: 프로젝트 개요


⚠️ 주의사항:
- 이 파일의 구조는 가이드일 뿐입니다
- 자유롭게 재설계하고 확장할 수 있습니다
- 단, generate_response() 함수 시그니처는 유지해야 합니다
  (app.py에서 호출하기 때문)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import json

# 환경변수 로드
load_dotenv()

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent


class ChatbotService:
    """
    챗봇 서비스 클래스
    
    이 클래스는 챗봇의 모든 AI 로직을 캡슐화합니다.
    
    주요 책임:
    1. OpenAI API 관리
    2. ChromaDB 벡터 검색
    3. LangChain 메모리 관리
    4. 응답 생성 파이프라인
    
    직접 구현해야 할 메서드:
    - __init__: 모든 구성 요소 초기화
    - _load_config: 설정 파일 로드
    - _init_chromadb: 벡터 데이터베이스 초기화
    - _create_embedding: 텍스트 → 벡터 변환
    - _search_similar: RAG 검색 수행 (핵심!)
    - _build_prompt: 프롬프트 구성
    - generate_response: 최종 응답 생성 (모든 로직 통합)
    """
    
    def __init__(self):
        """
        챗봇 서비스 초기화

        초기화 항목:
        1. Config 로드 (chatbot_config.json)
        2. OpenAI Client (임베딩용)
        3. ChromaDB (벡터 검색용)
        4. LangChain ChatOpenAI (응답 생성용)
        5. 메모리 스토어 (대화 기록 관리)
        """
        print("[ChatbotService] 초기화 중...")

        # 1. Config 로드
        self.config = self._load_config()
        print(f"[ChatbotService] Config 로드 완료: {self.config.get('name', 'Unknown')}")

        # 2. OpenAI Client 초기화 (임베딩용)
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        self.client = OpenAI(api_key=api_key)
        print("[ChatbotService] OpenAI Client 초기화 완료")

        # 3. ChromaDB 초기화
        try:
            self.collection = self._init_chromadb()
            print(f"[ChatbotService] ChromaDB 연결 완료: {self.collection.count()} 문서")
        except Exception as e:
            print(f"[ChatbotService] ChromaDB 연결 실패: {e}")
            print("[ChatbotService] RAG 검색 없이 계속 진행합니다.")
            self.collection = None

        # 4. LangChain ChatOpenAI 초기화 (응답 생성용)
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=500,
            api_key=api_key
        )
        print("[ChatbotService] LangChain ChatOpenAI 초기화 완료")

        # 5. 메모리 스토어 초기화 (세션별 대화 기록)
        # 최신 LangChain 방식: InMemoryChatMessageHistory 사용
        from langchain_core.chat_history import InMemoryChatMessageHistory
        self.message_store = {}  # session_id -> InMemoryChatMessageHistory
        print("[ChatbotService] 메모리 스토어 초기화 완료")

        print("[ChatbotService] 초기화 완료 ✅\n")
    
    
    def _load_config(self):
        """
        설정 파일 로드

        Returns:
            dict: 챗봇 설정 정보
        """
        config_path = BASE_DIR / 'config' / 'chatbot_config.json'
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"[WARNING] 설정 파일을 찾을 수 없습니다: {config_path}")
            # 기본 설정 반환
            return {
                'name': '챗봇',
                'description': '챗봇 설명',
                'tags': ['#챗봇'],
                'character': {},
                'system_prompt': {
                    'base': '당신은 친근한 AI 어시스턴트입니다.',
                    'rules': ['친절하게 대답하세요']
                }
            }
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON 파싱 오류: {e}")
            raise
    
    
    def _init_chromadb(self):
        """
        ChromaDB 초기화 및 컬렉션 반환

        Returns:
            chromadb.Collection: ChromaDB 컬렉션

        Raises:
            Exception: ChromaDB 연결 실패 시
        """
        import chromadb

        db_path = BASE_DIR / "static" / "data" / "chatbot" / "chardb_embedding"

        # 디렉토리가 없으면 생성
        db_path.mkdir(parents=True, exist_ok=True)

        # PersistentClient 생성
        client = chromadb.PersistentClient(path=str(db_path))

        # 컬렉션 가져오기 (없으면 생성)
        try:
            collection = client.get_collection(name="rag_collection")
        except Exception:
            # 컬렉션이 없으면 새로 생성
            print("[ChromaDB] 'rag_collection' 컬렉션이 없어 새로 생성합니다.")
            collection = client.create_collection(name="rag_collection")

        return collection
    
    
    def _create_embedding(self, text: str) -> list:
        """
        텍스트를 임베딩 벡터로 변환

        Args:
            text (str): 임베딩할 텍스트

        Returns:
            list: 3072차원 벡터 (text-embedding-3-large 모델)
        """
        try:
            response = self.client.embeddings.create(
                input=[text],
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[ERROR] 임베딩 생성 실패: {e}")
            raise
    
    
    def _search_similar(self, query: str, threshold: float = 0.45, top_k: int = 5):
        """
        RAG 검색: 유사한 문서 찾기 (핵심 메서드!)

        Args:
            query (str): 검색 질의
            threshold (float): 유사도 임계값 (0.3-0.5 권장)
            top_k (int): 검색할 문서 개수

        Returns:
            tuple: (document, similarity, metadata) 또는 (None, None, None)

        핵심 개념:
        - Distance vs Similarity
          · ChromaDB는 "거리(distance)"를 반환 (작을수록 유사)
          · 우리는 "유사도(similarity)"로 변환 (클수록 유사)
          · 변환 공식: similarity = 1 / (1 + distance)
        """
        # ChromaDB가 초기화되지 않은 경우
        if self.collection is None:
            return (None, None, None)

        try:
            # 1. 쿼리 임베딩 생성
            query_embedding = self._create_embedding(query)

            # 2. ChromaDB 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )

            # 검색 결과가 없는 경우
            if not results['documents'][0]:
                print(f"[RAG] 검색 결과 없음")
                return (None, None, None)

            # 3. 유사도 계산 및 필터링
            best_document = None
            best_similarity = 0
            best_metadata = None

            documents = results['documents'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]

            for doc, dist, meta in zip(documents, distances, metadatas):
                # 유사도 계산: similarity = 1 / (1 + distance)
                similarity = 1 / (1 + dist)

                # 디버깅 출력
                print(f"[RAG] 문서: {doc[:50]}... | 거리: {dist:.4f} | 유사도: {similarity:.4f}")

                # threshold 이상이고 현재까지 최고 유사도인 경우
                if similarity >= threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_document = doc
                    best_metadata = meta

            # 4. 최적 문서 반환
            if best_document:
                print(f"[RAG] ✅ 선택된 문서 유사도: {best_similarity:.4f}")
                return (best_document, best_similarity, best_metadata)
            else:
                print(f"[RAG] ❌ threshold({threshold}) 이상인 문서 없음")
                return (None, None, None)

        except Exception as e:
            print(f"[ERROR] RAG 검색 실패: {e}")
            return (None, None, None)
    
    
    def _build_prompt(self, user_message: str, context: str = None, username: str = "사용자"):
        """
        LLM 프롬프트 구성

        Args:
            user_message (str): 사용자 메시지
            context (str): RAG 검색 결과 (선택)
            username (str): 사용자 이름

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        # 1. 시스템 프롬프트 구성
        system_prompt_config = self.config.get('system_prompt', {})
        base_prompt = system_prompt_config.get('base', '당신은 친근한 AI 어시스턴트입니다.')
        rules = system_prompt_config.get('rules', [])

        system_prompt = base_prompt
        if rules:
            system_prompt += "\n\n[대화 규칙]\n" + "\n".join(f"- {rule}" for rule in rules)

        # 2. 사용자 프롬프트 구성
        user_prompt = ""

        # RAG 컨텍스트 추가
        if context:
            user_prompt += f"[참고 정보]\n{context}\n\n"

        # 사용자 메시지 추가
        user_prompt += f"{username}: {user_message}"

        return (system_prompt, user_prompt)
    
    
    def generate_response(self, user_message: str, username: str = "사용자", session_id: str = "default") -> dict:
        """
        사용자 메시지에 대한 챗봇 응답 생성 (LangChain 사용)

        Args:
            user_message (str): 사용자 입력
            username (str): 사용자 이름
            session_id (str): 세션 ID (대화 기록 관리용)

        Returns:
            dict: {
                'reply': str,       # 챗봇 응답 텍스트
                'image': str|None   # 이미지 경로 (선택)
            }
        """
        try:
            print(f"\n{'='*60}")
            print(f"[USER] {username} (session: {session_id}): {user_message}")

            # [1단계] 초기 메시지 처리
            if user_message.strip().lower() == "init":
                bot_name = self.config.get('name', '챗봇')
                description = self.config.get('description', '')
                init_message = f"안녕! 나는 {bot_name}이야."
                if description:
                    init_message += f"\n{description}"

                print(f"[BOT] {init_message}")
                print(f"{'='*60}\n")
                return {
                    'reply': init_message,
                    'image': None
                }

            # [2단계] RAG 검색 수행
            context, similarity, metadata = self._search_similar(
                query=user_message,
                threshold=0.45,
                top_k=5
            )
            has_context = (context is not None)

            # [3단계] 프롬프트 구성
            system_prompt, user_prompt = self._build_prompt(
                user_message=user_message,
                context=context,
                username=username
            )

            print(f"[RAG] Context found: {has_context}")
            if has_context:
                print(f"[RAG] Similarity: {similarity:.4f}")
                print(f"[RAG] Context preview: {context[:100]}...")

            # [4단계] LangChain으로 LLM 호출
            # 메모리가 있는 경우와 없는 경우 분기
            from langchain_core.messages import SystemMessage, HumanMessage
            from langchain_core.chat_history import InMemoryChatMessageHistory
            from langchain_core.runnables.history import RunnableWithMessageHistory

            # 세션별 메모리 가져오기 또는 생성
            if session_id not in self.message_store:
                self.message_store[session_id] = InMemoryChatMessageHistory()

            session_history = self.message_store[session_id]

            # 메시지 구성
            messages = [SystemMessage(content=system_prompt)]

            # 대화 기록 추가
            messages.extend(session_history.messages)

            # 현재 사용자 메시지 추가
            messages.append(HumanMessage(content=user_prompt))

            print(f"[LLM] Calling ChatOpenAI... (대화 기록: {len(session_history.messages)}개)")

            # LLM 호출
            response = self.llm.invoke(messages)
            reply = response.content

            print(f"[BOT] {reply}")
            print(f"{'='*60}\n")

            # [5단계] 메모리 저장
            session_history.add_user_message(user_prompt)
            session_history.add_ai_message(reply)

            # [6단계] 응답 반환
            return {
                'reply': reply,
                'image': None  # 이미지 검색 로직은 추후 추가 가능
            }

        except Exception as e:
            import traceback
            print(f"[ERROR] 응답 생성 실패: {e}")
            print(traceback.format_exc())
            return {
                'reply': "죄송해요, 일시적인 오류가 발생했어요. 다시 시도해주세요.",
                'image': None
            }


# ============================================================================
# 싱글톤 패턴
# ============================================================================
# ChatbotService 인스턴스를 앱 전체에서 재사용
# (매번 새로 초기화하면 비효율적)

_chatbot_service = None

def get_chatbot_service():
    """
    챗봇 서비스 인스턴스 반환 (싱글톤)
    
    첫 호출 시 인스턴스 생성, 이후 재사용
    """
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service


# ============================================================================
# 테스트용 메인 함수
# ============================================================================

if __name__ == "__main__":
    """
    로컬 테스트용
    
    실행 방법:
    python services/chatbot_service.py
    """
    print("챗봇 서비스 테스트")
    print("=" * 50)
    
    service = get_chatbot_service()
    
    # 초기화 테스트
    response = service.generate_response("init", "테스터")
    print(f"초기 응답: {response}")
    
    # 일반 대화 테스트
    response = service.generate_response("안녕하세요!", "테스터")
    print(f"응답: {response}")
