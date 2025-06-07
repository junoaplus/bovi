# 모델 업데이트 요약

## 변경 사항

### 🔄 파인튜닝 모델 교체
- **기존**: `beomi/KoAlpaca-Polyglot-5.8B` + LoRA 어댑터 (`./models/koalpaca-bang-model`)
- **변경**: `minjeongHuggingFace/koalpaca-bang_e9` (허깅페이스에서 직접 로드)

### 📁 수정된 파일들

#### 1. `services/finetuning_service.py`
- **주요 변경사항**:
  - 허깅페이스 모델 `minjeongHuggingFace/koalpaca-bang_e9` 직접 로드
  - LoRA/PEFT 로직 제거 (완전 파인튜닝된 모델이므로)
  - 오류 시 기본 모델 `beomi/KoAlpaca-Polyglot-5.8B`로 폴백
  - `peft` 라이브러리 import 제거

#### 2. `requirements.txt`
- **변경사항**:
  - `peft==0.15.2` 라이브러리 제거 (더 이상 필요하지 않음)
  - 주석 업데이트: "LoRA/PEFT" → "모델 가속화"

#### 3. `.env`
- **변경사항**:
  - `FINETUNING_MODEL_ID`: `ft:gpt-3.5-turbo-0125:tset::BX2RnWfq` → `minjeongHuggingFace/koalpaca-bang_e9`

#### 4. `.env.example`
- **변경사항**:
  - `FINETUNING_MODEL_ID` 예시값 업데이트

### 🚀 사용법

#### 기존 설치된 환경에서 업데이트
```bash
# peft 라이브러리 제거 (선택사항)
pip uninstall peft

# 새로운 requirements.txt 설치
pip install -r requirements.txt
```

#### 새로 설치하는 경우
```bash
git clone <repository-url>
cd runpod_ai_backend
pip install -r requirements.txt
python main.py
```

### 🔧 작동 방식

1. **1차 시도**: `minjeongHuggingFace/koalpaca-bang_e9` 모델 로드
2. **실패 시**: `beomi/KoAlpaca-Polyglot-5.8B` 기본 모델로 폴백
3. **완전 실패 시**: 모델 없이 작동 (RAG 서비스만 사용)

### ✅ 장점

- **간단한 모델 로드**: LoRA 어댑터 없이 완전 파인튜닝된 모델 직접 사용
- **안정성 향상**: 허깅페이스에서 직접 로드하여 버전 관리 용이
- **폴백 시스템**: 모델 로드 실패 시 기본 모델로 자동 대체
- **의존성 감소**: PEFT 라이브러리 제거로 설치 복잡도 감소

### 📋 테스트 방법

서버 시작 후 다음 엔드포인트로 테스트:

```bash
# 헬스체크
curl http://localhost:8000/health

# 파인튜닝 모델로 룰 설명 테스트
curl -X POST http://localhost:8000/explain-rules \
  -H "Content-Type: application/json" \
  -d '{
    "game_name": "방 탈출",
    "question": "게임 시작은 어떻게 하나요?",
    "chat_type": "finetuning"
  }'
```

### 🔍 로그 확인 사항

서버 시작 시 다음과 같은 로그가 나타나야 합니다:

```
📥 파인튜닝된 모델 로드 중: minjeongHuggingFace/koalpaca-bang_e9
📥 Tokenizer 로드 중...
✅ 파인튜닝 모델 로드 완료
```

실패 시:
```
❌ 모델 로드 실패: [오류 메시지]
🔄 기본 모델로 대체 시도...
📥 기본 모델 로드: beomi/KoAlpaca-Polyglot-5.8B
✅ 기본 모델 로드 완료
```

### 🚨 주의사항

- 첫 실행 시 허깅페이스에서 모델을 다운로드하므로 인터넷 연결과 충분한 저장공간이 필요합니다
- GPU 메모리가 부족한 경우 `torch_dtype=torch.float16` 및 `device_map="auto"` 설정이 도움이 됩니다
- 모델 크기가 큰 경우 로딩에 시간이 걸릴 수 있습니다

---

**업데이트 완료일**: 2025년 6월 7일  
**업데이트 담당자**: AI Assistant  
**모델 소유자**: minjeongHuggingFace
