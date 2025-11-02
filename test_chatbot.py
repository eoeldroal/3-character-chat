#!/usr/bin/env python3
"""
챗봇 서비스 테스트 스크립트

이 스크립트는 chatbot_service.py 구현을 테스트합니다.
환경변수 설정 후 실행하세요:

    export OPENAI_API_KEY=your_api_key_here
    python test_chatbot.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 프로젝트 루트 경로 추가
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def test_environment():
    """환경 설정 테스트"""
    print("=" * 60)
    print("1. 환경 설정 테스트")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   .env 파일을 생성하고 API 키를 설정해주세요.")
        return False

    print(f"✅ OPENAI_API_KEY 설정됨: {api_key[:10]}...{api_key[-4:]}")
    return True


def test_imports():
    """필수 라이브러리 임포트 테스트"""
    print("\n" + "=" * 60)
    print("2. 라이브러리 임포트 테스트")
    print("=" * 60)

    required_packages = {
        'openai': 'OpenAI',
        'chromadb': 'ChromaDB',
        'langchain': 'LangChain Core',
        'langchain_openai': 'LangChain OpenAI',
        'flask': 'Flask',
        'dotenv': 'python-dotenv'
    }

    all_imported = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} 임포트 성공")
        except ImportError as e:
            print(f"❌ {name} 임포트 실패: {e}")
            all_imported = False

    return all_imported


def test_config():
    """설정 파일 테스트"""
    print("\n" + "=" * 60)
    print("3. 설정 파일 테스트")
    print("=" * 60)

    config_path = BASE_DIR / "config" / "chatbot_config.json"

    if not config_path.exists():
        print(f"❌ 설정 파일이 없습니다: {config_path}")
        return False

    print(f"✅ 설정 파일 존재: {config_path}")

    import json
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        required_keys = ['name', 'description', 'system_prompt']
        for key in required_keys:
            if key in config:
                print(f"✅ '{key}' 키 존재")
            else:
                print(f"⚠️  '{key}' 키 없음 (선택 사항)")

        return True
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        return False


def test_chromadb_setup():
    """ChromaDB 디렉토리 확인"""
    print("\n" + "=" * 60)
    print("4. ChromaDB 디렉토리 테스트")
    print("=" * 60)

    db_path = BASE_DIR / "static" / "data" / "chatbot" / "chardb_embedding"

    if db_path.exists():
        print(f"✅ ChromaDB 디렉토리 존재: {db_path}")

        # 파일 개수 확인
        files = list(db_path.glob("**/*"))
        print(f"   파일/폴더 개수: {len(files)}")
    else:
        print(f"⚠️  ChromaDB 디렉토리 없음 (자동 생성됩니다): {db_path}")
        print("   첫 실행 시 자동으로 생성됩니다.")

    return True


def test_chatbot_service_init():
    """ChatbotService 초기화 테스트"""
    print("\n" + "=" * 60)
    print("5. ChatbotService 초기화 테스트")
    print("=" * 60)

    try:
        from services import get_chatbot_service

        print("ChatbotService 초기화 중...")
        chatbot = get_chatbot_service()

        print("✅ ChatbotService 초기화 성공")

        # 속성 확인
        print("\n[초기화된 구성요소]")
        print(f"  - Config: {'✅' if chatbot.config else '❌'}")
        print(f"  - OpenAI Client: {'✅' if chatbot.client else '❌'}")
        print(f"  - ChromaDB Collection: {'✅' if chatbot.collection else '⚠️  (비어있음)'}")
        print(f"  - LangChain Memory: {'✅' if chatbot.memory else '⚠️  (선택사항)'}")

        if chatbot.config:
            print(f"\n[챗봇 설정]")
            print(f"  - 이름: {chatbot.config.get('name', 'Unknown')}")

        return True

    except Exception as e:
        print(f"❌ ChatbotService 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding():
    """임베딩 생성 테스트"""
    print("\n" + "=" * 60)
    print("6. 임베딩 생성 테스트")
    print("=" * 60)

    try:
        from services import get_chatbot_service
        chatbot = get_chatbot_service()

        test_text = "안녕하세요"
        print(f"테스트 텍스트: '{test_text}'")
        print("임베딩 생성 중...")

        embedding = chatbot._create_embedding(test_text)

        print(f"✅ 임베딩 생성 성공")
        print(f"   벡터 차원: {len(embedding)}")
        print(f"   벡터 샘플: [{embedding[0]:.4f}, {embedding[1]:.4f}, ..., {embedding[-1]:.4f}]")

        return True

    except Exception as e:
        print(f"❌ 임베딩 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generate_response():
    """응답 생성 테스트"""
    print("\n" + "=" * 60)
    print("7. 응답 생성 테스트")
    print("=" * 60)

    try:
        from services import get_chatbot_service
        chatbot = get_chatbot_service()

        # 초기 메시지 테스트
        print("\n[테스트 1] 초기 인사 메시지")
        response1 = chatbot.generate_response("init", "테스터")
        print(f"✅ 응답: {response1['reply']}")

        # 일반 대화 테스트
        print("\n[테스트 2] 일반 대화")
        response2 = chatbot.generate_response("안녕하세요!", "테스터")
        print(f"✅ 응답: {response2['reply'][:100]}...")

        return True

    except Exception as e:
        print(f"❌ 응답 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "챗봇 서비스 테스트" + " " * 22 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    tests = [
        ("환경 설정", test_environment),
        ("라이브러리 임포트", test_imports),
        ("설정 파일", test_config),
        ("ChromaDB 디렉토리", test_chromadb_setup),
        ("ChatbotService 초기화", test_chatbot_service_init),
        ("임베딩 생성", test_embedding),
        ("응답 생성", test_generate_response),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except KeyboardInterrupt:
            print("\n\n테스트가 중단되었습니다.")
            sys.exit(1)
        except Exception as e:
            print(f"\n예상치 못한 오류 발생: {e}")
            results.append((name, False))

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status} - {name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"전체: {passed}/{total} 테스트 통과")
    print("=" * 60)

    if passed == total:
        print("\n🎉 모든 테스트가 통과했습니다!")
        print("챗봇 서비스가 정상적으로 작동합니다.")
    else:
        print("\n⚠️  일부 테스트가 실패했습니다.")
        print("위의 오류 메시지를 확인하고 문제를 해결해주세요.")

    print()


if __name__ == "__main__":
    main()
