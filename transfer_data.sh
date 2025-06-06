#!/bin/bash

# Runpod으로 데이터 전송 스크립트

echo "📦 Runpod으로 AI 데이터를 전송합니다..."

# 데이터 디렉토리 확인
DATA_DIR="../play/data"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 데이터 디렉토리를 찾을 수 없습니다: $DATA_DIR"
    exit 1
fi

# Runpod 인스턴스 IP 입력 받기
read -p "Runpod 인스턴스 IP 또는 도메인을 입력하세요: " RUNPOD_IP

if [ -z "$RUNPOD_IP" ]; then
    echo "❌ Runpod IP가 입력되지 않았습니다."
    exit 1
fi

# SSH 키 경로 입력 받기
read -p "SSH 키 파일 경로를 입력하세요 (예: ~/.ssh/runpod_key): " SSH_KEY

if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH 키 파일을 찾을 수 없습니다: $SSH_KEY"
    exit 1
fi

# 데이터 압축
echo "🗜️ 데이터를 압축합니다..."
cd $DATA_DIR
tar -czf ai_data.tar.gz *
cd - > /dev/null

# Runpod으로 파일 전송
echo "📤 파일을 전송합니다..."
scp -i $SSH_KEY $DATA_DIR/ai_data.tar.gz root@$RUNPOD_IP:/app/

# Runpod에서 압축 해제
echo "📂 Runpod에서 파일을 압축 해제합니다..."
ssh -i $SSH_KEY root@$RUNPOD_IP << 'EOF'
cd /app
mkdir -p data
tar -xzf ai_data.tar.gz -C data/
rm ai_data.tar.gz
echo "✅ 데이터 압축 해제 완료"
ls -la data/
EOF

# 로컬 압축 파일 정리
rm $DATA_DIR/ai_data.tar.gz

echo "✅ 데이터 전송이 완료되었습니다!"
echo "🚀 이제 Runpod에서 AI 서버를 시작할 수 있습니다."
