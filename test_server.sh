#!/bin/bash

# runpod_ai_backend 테스트 스크립트

echo "🚀 보드게임 AI 백엔드 테스트를 시작합니다..."

# 현재 디렉토리를 runpod_ai_backend로 변경
cd /Users/hwangjunho/Desktop/project/runpod_ai_backend

echo "📋 환경변수 확인..."
if [ -f .env ]; then
    echo "✅ .env 파일 존재"
    echo "🔑 OPENAI_API_KEY=$(head -c 20 .env | grep OPENAI_API_KEY | cut -d'=' -f2 | head -c 20)..."
else
    echo "❌ .env 파일 없음"
    exit 1
fi

echo "📦 필요한 Python 패키지 확인..."
python3 -c "import fastapi, uvicorn, sentence_transformers, faiss, numpy, langchain_openai; print('✅ 필수 패키지 모두 설치됨')" || {
    echo "❌ 필수 패키지가 설치되지 않았습니다. pip install -r requirements.txt를 실행하세요."
    exit 1
}

echo "🗂️ 데이터 파일 확인..."
data_files=("data/game_names.json" "data/texts.json" "data/game_index.faiss" "data/game.json")
for file in "${data_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file 존재"
    else
        echo "⚠️ $file 없음"
    fi
done

echo "🧪 간단한 API 테스트 서버 실행..."
echo "다음 명령어로 서버를 실행하세요:"
echo "cd /Users/hwangjunho/Desktop/project/runpod_ai_backend"
echo "python3 main_simple.py"
echo ""
echo "또는 완전한 서버:"
echo "python3 main.py"
echo ""
echo "서버 실행 후 다음 URL들을 테스트하세요:"
echo "- http://localhost:8000/ (루트)"
echo "- http://localhost:8000/health (헬스체크)"  
echo "- http://localhost:8000/games (게임 목록)"
echo "- http://localhost:8000/docs (API 문서)"
