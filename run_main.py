#!/usr/bin/env python3
"""
RAG Playground 메인 서버 실행 스크립트
- main.py를 uvicorn으로 실행
- 개발 및 프로덕션 환경 지원
- conda 가상환경 자동 활성화 지원
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_wsl_environment():
    """WSL 환경 확인"""
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read()
            if 'Microsoft' in version_info or 'WSL' in version_info:
                print("[INFO] WSL 환경에서 실행 중입니다.")
                return True
            else:
                print("[INFO] 일반 Linux 환경에서 실행 중입니다.")
                return False
    except FileNotFoundError:
        print("[INFO] Windows 환경에서 실행 중입니다.")
        return False

def check_venv_environment():
    """Python 가상환경 확인"""
    print("[INFO] Python 가상환경 확인 중...")
    
    # .venv 디렉토리 확인
    venv_path = Path(".venv")
    if venv_path.exists():
        print("[SUCCESS] .venv 가상환경이 존재합니다.")
        return True
    else:
        print("[WARNING] .venv 가상환경이 없습니다.")
        print("   다음 명령어로 생성하세요:")
        print("   python -m venv .venv")
        print("   # Windows:")
        print("   .venv\\Scripts\\activate")
        print("   # Linux/Mac:")
        print("   source .venv/bin/activate")
        print("   pip install -r requirements.txt")
        return False

def check_dependencies():
    """필수 의존성 확인"""
    print("[INFO] 의존성 확인 중...")
    
    # 기본 패키지만 확인 (import 에러 방지)
    basic_packages = ['uvicorn', 'fastapi']
    missing_packages = []
    
    for package in basic_packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [MISSING] {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[ERROR] 누락된 패키지: {', '.join(missing_packages)}")
        print("   다음 명령어로 설치하세요:")
        print("   conda activate llm_test")
        print("   pip install -r requirements.txt")
        return False
    
    print("[SUCCESS] 기본 의존성이 설치되어 있습니다.")
    print("[INFO] 추가 의존성은 서버 실행 시 확인됩니다.")
    return True

def check_environment():
    """환경 설정 확인"""
    env_file = Path(".env")
    if not env_file.exists():
        print("[WARNING] .env 파일이 없습니다.")
        print("   .env 파일을 생성하고 필요한 환경 변수를 설정하세요.")
        return False
    
    print("[SUCCESS] .env 파일이 존재합니다.")
    return True

def check_ssl_certificates():
    """SSL 인증서 파일 확인"""
    key_file = Path("key.pem")
    cert_file = Path("cert.pem")
    
    if key_file.exists() and cert_file.exists():
        print("[SUCCESS] SSL 인증서 파일이 존재합니다.")
        return True
    else:
        print("[WARNING] SSL 인증서 파일이 없습니다.")
        print("   HTTPS를 사용하려면 key.pem과 cert.pem 파일이 필요합니다.")
        return False

def check_main_file():
    """main.py 파일 확인"""
    main_file = Path("main.py")
    if not main_file.exists():
        print("[ERROR] main.py 파일을 찾을 수 없습니다.")
        return False
    
    print("[SUCCESS] main.py 파일이 존재합니다.")
    return True

def run_in_venv(cmd):
    """Python 가상환경에서 명령어 실행"""
    if os.name == 'nt':  # Windows
        venv_cmd = f".venv\\Scripts\\activate && {' '.join(cmd)}"
    else:  # Linux/Mac
        venv_cmd = f"source .venv/bin/activate && {' '.join(cmd)}"
    
    return subprocess.run(venv_cmd, shell=True)

def run_server(host="0.0.0.0", port=5001, reload=False, check_only=False, use_venv=True, use_ssl=False):
    """서버 실행"""
    if check_only:
        print("환경 확인만 수행합니다.")
        return
    
    # uvicorn 명령어 구성
    cmd = [
        "uvicorn", 
        "main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    if use_ssl:
        cmd.extend(["--ssl-keyfile=./key.pem", "--ssl-certfile=./cert.pem"])
    
    protocol = "https" if use_ssl else "http"
    print(f"서버를 시작합니다...")
    print(f"  호스트: {host}")
    print(f"  포트: {port}")
    print(f"  프로토콜: {protocol.upper()}")
    print(f"  리로드: {'활성화' if reload else '비활성화'}")
    print(f"  가상환경: {'사용' if use_venv else '미사용'}")
    print(f"  URL: {protocol}://{host}:{port}")
    print(f"  명령어: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        if use_venv:
            print("[INFO] .venv 가상환경에서 실행합니다.")
            run_in_venv(cmd)
        else:
            print("[INFO] 현재 환경에서 실행합니다.")
            subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] 서버가 중지되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] 서버 실행 중 오류가 발생했습니다: {e}")
        if use_venv:
            print("\n[HELP] 가상환경을 수동으로 활성화하려면:")
            print("   # Windows:")
            print("   .venv\\Scripts\\activate")
            print("   # Linux/Mac:")
            print("   source .venv/bin/activate")
            print("   python run_main.py --no-venv")
        sys.exit(1)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="RAG Playground 메인 서버 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_main.py                    # 기본 실행 (포트 5001)
  python run_main.py --port 8080       # 포트 변경
  python run_main.py --reload          # 개발 모드 (자동 리로드)
  python run_main.py --ssl             # SSL로 실행 (https://0.0.0.0:5001)
  python run_main.py --check-only      # 환경 확인만
  python run_main.py --host 127.0.0.1  # 호스트 변경
  python run_main.py --no-venv         # 가상환경 사용 안함
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
        default=5001,
        help="서버 포트 (기본값: 5001)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="개발 모드 활성화 (자동 리로드)"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="환경 확인만 하고 종료"
    )
    
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="가상환경 사용 안함 (현재 환경에서 실행)"
    )
    
    parser.add_argument(
        "--ssl",
        action="store_true",
        help="SSL로 서버 실행 (cert.pem, key.pem 필요)"
    )
    
    args = parser.parse_args()
    
    print("RAG Playground 서버 실행 스크립트")
    print("=" * 50)
    
    # 환경 확인
    print("1. 환경 확인 중...")
    
    # 가상환경 확인
    use_venv = not args.no_venv
    if use_venv:
        venv_available = check_venv_environment()
        if not venv_available:
            print("[WARNING] 가상환경을 사용할 수 없습니다. 현재 환경에서 실행합니다.")
            use_venv = False
    
    if not check_dependencies():
        if not args.check_only:
            sys.exit(1)
    
    if not check_environment():
        if not args.check_only:
            print("[WARNING] .env 파일이 없지만 계속 진행합니다.")
    
    if not check_main_file():
        sys.exit(1)
    
    # SSL 옵션 확인
    if args.ssl:
        if not check_ssl_certificates():
            print("[ERROR] SSL을 사용하려면 cert.pem과 key.pem 파일이 필요합니다.")
            sys.exit(1)
    
    if args.check_only:
        print("\n[SUCCESS] 환경 확인 완료!")
        print(f"\n다음 명령어로 서버를 시작할 수 있습니다:")
        print(f"  python run_main.py --host {args.host} --port {args.port}")
        if args.reload:
            print("  python run_main.py --reload")
        if not use_venv:
            print("  python run_main.py --no-venv")
        return
    
    # 서버 실행
    print("\n2. 서버 시작 중...")
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        check_only=args.check_only,
        use_venv=use_venv,
        use_ssl=args.ssl
    )

if __name__ == "__main__":
    main()
