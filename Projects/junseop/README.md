AI 공정 전문가 대시보드: 실행 가이드

1. 사전 설정 (Initial Setup)
각 디렉토리에서 필요한 라이브러리를 설치합니다.

백엔드 (Backend)
bash
# 1. 백엔드 폴더로 이동
cd backend

# 2. 파이썬 가상 환경 생성
python -m venv venv

# 3. 가상 환경 활성화
# Windows에서 아래 activate 명령어 실행 시 권한 오류가 발생할 경우,
# 바로 아래의 보안 정책 변경 명령어를 먼저 실행한 후 다시 시도하세요.

venv\Scripts\activate


❗ Windows PowerShell 오류 발생 시

venv\Scripts\activate 실행 시 ...이 시스템에서 스크립트를 실행할 수 없으므로... 와 같은 오류가 발생하면, 아래 명령어를 먼저 실행하여 현재 터미널의 실행 정책을 임시로 변경해주세요.


powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

bash
# 4. 필요 라이브러리 설치 (가상 환경 활성화 후)
pip install -r requirements.txt
프론트엔드 (Frontend)

bash
# 1. 프론트엔드 폴더로 이동
cd frontend

# 2. 필요 라이브러리 설치
npm install

2. 서버 실행 (Running Servers)
백엔드와 프론트엔드 서버는 각각 다른 터미널에서 실행해야 합니다.

백엔드 서버 실행
첫 번째 터미널에서 아래 명령어를 실행합니다.

bash
# 1. backend 폴더로 이동
cd backend

# 2. 가상 환경 활성화
venv\Scripts\activate

# 3. 서버 실행
python app.py
프론트엔드 서버 실행
두 번째 터미널에서 아래 명령어를 실행합니다.

bash
# 1. frontend 폴더로 이동
cd frontend

# 2. 서버 실행
npm start
3. 애플리케이션 확인
프론트엔드 서버 실행 후, 자동으로 열리는 웹 브라우저(http://localhost:3000)에서 대시보드를 확인할 수 있습니다.