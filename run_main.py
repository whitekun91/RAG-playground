#!/usr/bin/env python3
"""
RAG Playground 메인 실행 스크립트
- FastAPI 기반 RAG 서버 실행
- 환경 설정 및 의존성 확인
- 자동 서버 시작 및 관리
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """필수 의존성 확인"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("[ERROR] 누락된 패키지들:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("[SUCCESS] 기본 필수 패키지가 설치되어 있습니다.")
    print("[INFO] 추가 패키지는 필요시 자동으로 설치됩니다.")
    return True

def check_environment():
    """환경 변수 및 설정 확인"""
    print("환경 설정 확인 중...")
    
    # .env 파일 확인
    env_file = Path(".env")
    if not env_file.exists():
        print("[WARNING] .env 파일이 없습니다. 기본 설정을 사용합니다.")
        print("   필요시 .env 파일을 생성하여 환경 변수를 설정하세요.")
    else:
        print("[SUCCESS] .env 파일이 존재합니다.")
    
    # 모델 경로 확인
    model_paths = [
        "core/models/embeddings/ko-sbert-sts",
        "core/models/embeddings/bge-reranker-v2-m3",
        "core/models/STT_Models/whisper-large-v3-turbo",
        "core/models/TTS_Models/bark"
    ]
    
    missing_models = []
    for model_path in model_paths:
        if not Path(model_path).exists():
            missing_models.append(model_path)
    
    if missing_models:
        print("[WARNING] 일부 모델이 없습니다:")
        for model in missing_models:
            print(f"   - {model}")
        print("   모델이 없어도 기본 기능은 작동합니다.")
    else:
        print("[SUCCESS] 모든 모델이 준비되어 있습니다.")
    
    # 벡터 DB 확인
    vector_db_path = Path("documents/vector_db")
    if vector_db_path.exists():
        print("[SUCCESS] 벡터 데이터베이스가 준비되어 있습니다.")
    else:
        print("[WARNING] 벡터 데이터베이스가 없습니다.")
        print("   PDF 문서를 업로드하여 벡터 DB를 생성하세요.")
    
    return True

def start_server(host="0.0.0.0", port=8000, reload=False):
    """FastAPI 서버 시작"""
    print(f"RAG Playground 서버를 시작합니다...")
    print(f"   호스트: {host}")
    print(f"   포트: {port}")
    print(f"   리로드: {'활성화' if reload else '비활성화'}")
    print()
    
    # uvicorn 명령어 구성
    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    # 개발 모드 추가 옵션
    if reload:
        cmd.extend([
            "--reload-dir", ".",
            "--reload-exclude", "*.pyc",
            "--reload-exclude", "__pycache__",
            "--reload-exclude", "*.log"
        ])
    
    try:
        print("서버 시작 중...")
        print("   브라우저에서 http://localhost:8000 을 열어주세요.")
        print("   서버를 중지하려면 Ctrl+C를 누르세요.")
        print("-" * 50)
        
        # 서버 실행
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n[INFO] 서버가 중지되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 서버 시작 중 오류가 발생했습니다: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 예상치 못한 오류가 발생했습니다: {e}")
        return False
    
    return True

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="RAG Playground 메인 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_main.py                       # 기본 설정으로 서버 시작
  python run_main.py --port 8080          # 포트 8080으로 서버 시작
  python run_main.py --host 127.0.0.1     # 로컬호스트로 서버 시작
  python run_main.py --reload             # 개발 모드 (자동 리로드)
  python run_main.py --check-only         # 의존성만 확인하고 종료
        """
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="서버 호스트 (기본값: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="서버 포트 (기본값: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="개발 모드 활성화 (자동 리로드)"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="의존성 및 환경만 확인하고 종료"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="의존성 및 환경 확인 건너뛰기"
    )
    
    args = parser.parse_args()
    
    print("RAG Playground 통합 실행 스크립트")
    print("=" * 50)
    
    # 의존성 및 환경 확인
    if not args.skip_checks:
        print("1. 의존성 확인 중...")
        if not check_dependencies():
            print("\n[ERROR] 의존성 확인 실패. 서버를 시작할 수 없습니다.")
            sys.exit(1)
        
        print("\n2. 환경 설정 확인 중...")
        if not check_environment():
            print("\n[ERROR] 환경 설정 확인 실패.")
            sys.exit(1)
        
        print("\n[SUCCESS] 모든 확인이 완료되었습니다!")
    
    # 체크만 하고 종료
    if args.check_only:
        print("\n[SUCCESS] 체크 완료! 서버를 시작하려면 --check-only 옵션을 제거하세요.")
        return
    
    # 서버 시작
    print("\n3. 서버 시작 중...")
    success = start_server(
        host=args.host,
        port=args.port,
        reload=args.reload
    )
    
    if success:
        print("\n[SUCCESS] 서버가 정상적으로 종료되었습니다.")
    else:
        print("\n[ERROR] 서버 실행 중 오류가 발생했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()
