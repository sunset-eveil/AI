# Eveil AI

한국어 텍스트에서 개인정보의 원문 위치와 종류를 반환하는 KLUE-RoBERTa 기반 NER 프로젝트입니다. 학습·예측·검증 코드와 로컬 HTTP API를 포함합니다. 마스킹, 대응표 DB, 복원은 백엔드에서 처리합니다.

## 실행 환경

개발 환경은 Windows, Python 3.10.11, CPU용 PyTorch입니다. requirements.txt는 확인된 핵심 라이브러리 버전을 고정하며 CPU wheel 저장소를 추가로 사용합니다. 아래 명령은 프로젝트 루트에서 실행합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

.venv는 개발자별로 새로 만들고 Git으로 공유하지 않습니다. CPU 외 환경에서는 환경에 맞는 PyTorch 설치가 별도로 필요합니다.

## 학습 모델 준비: 필수

학습 가중치는 약 440 MB이므로 이 저장소에 포함하지 않습니다. 모델 담당자에게 **models/Eveil-AI 폴더 전체**를 별도로 전달받아 아래 위치에 배치해야 합니다. 다운로드 URL은 아직 없습니다. 코드만 clone한 상태에서는 추론 서버가 실행되지 않습니다.

```text
models/
  Eveil-AI/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
```

모델과 토크나이저는 local_files_only=True로 로드합니다. 원격에서 기본 모델을 받는 download_model.py는 models/base를 준비하는 코드이며, 현재 학습한 Eveil-AI 가중치를 대신하지 않습니다.

## HTTP 서버 실행

```powershell
.\.venv\Scripts\python.exe server.py
```

- 주소: http://127.0.0.1:8000
- API: POST /api/v1/pii/detect
- 같은 PC에서 실행한 백엔드에서만 접속할 수 있습니다. 외부 공개 배포를 위한 서버가 아닙니다.
- 모델은 서버 시작 시 한 번 로드합니다. 모델 로딩이 끝나기 전에는 포트를 열지 않습니다.
- 서버 종료: Ctrl+C. 모델 파일 변경 후에는 서버를 재시작해야 합니다.

## 요청과 응답

Content-Type: application/json; charset=utf-8

요청:

```json
{"request_id":"req-001","text":"😀 유나\n유나에게 전달해 주세요."}
```

탐지 결과 형식 예시:

```json
{
  "request_id": "req-001",
  "model_version": "eveil-ai-v1",
  "offset_unit": "unicode_codepoint",
  "entities": [
    {"start": 2, "end": 4, "type": "NAME"},
    {"start": 5, "end": 7, "type": "NAME"}
  ]
}
```

start는 포함, end는 제외하며 0부터 셉니다. 인덱스는 바이트나 토큰이 아닌 **Python 문자열의 유니코드 코드 포인트** 기준입니다. 이모지·결합문자 등은 Java/JavaScript의 UTF-16 인덱스와 다를 수 있으므로 변환 후 substring 처리해야 합니다. 원문 공백·줄바꿈을 변경하면 위치가 달라집니다.

같은 값이 여러 번 나오면 위치별로 각각 반환합니다. 같은 대체 표시를 재사용하는 작업은 백엔드가 처리합니다. 개인정보가 탐지되지 않으면 HTTP 200과 entities: []를 반환합니다. model_version은 현재 코드의 고정 식별자이며 학습 실행 이력을 자동으로 추적하지 않습니다.

유형: NAME(이름), PHONE(전화번호), ADDRESS(주소), ACCOUNT(계좌번호), CARD(카드번호), RRN(주민등록번호), EMAIL(이메일). OTHER는 사용자 추가 마스킹용이며 AI가 반환하지 않습니다. BIO 라벨은 추론 내부에서 범위로 합칩니다.

## PowerShell 호출 예시

서버를 켜 둔 상태에서 다른 터미널에서 실행합니다.

```powershell
$body = @{ request_id = "req-001"; text = "유나에게 전달해 주세요." } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/pii/detect" -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) | ConvertTo-Json -Depth 5
```

## 제한과 오류

입력은 특수 토큰 포함 256토큰 이하, HTTP 본문은 65,536바이트 이하입니다. 초과한 문장은 조용히 잘라 처리하지 않습니다. 동시에 한 건만 추론하고 이미 처리 중인 경우 기다리지 않고 오류를 반환합니다.

| HTTP | error.code | 의미 |
|---|---|---|
| 400 | INVALID_INPUT | JSON 오류, 필수 필드 누락, 타입 오류, 빈 입력 |
| 404 | NOT_FOUND | POST 요청의 API 경로가 잘못됨 |
| 413 | INPUT_TOO_LONG | 토큰 또는 본문 크기 한도 초과 |
| 503 | SERVER_BUSY | 다른 요청 추론 중 |
| 500 | INFERENCE_FAILED | 추론 오류 |

```json
{"request_id":"req-001","error":{"code":"INPUT_TOO_LONG","message":"256토큰을 초과했습니다."}}
```

본문 크기 초과 등 요청 ID를 읽기 전 발생하는 오류는 request_id: null을 반환합니다. 모델 로딩 실패나 서버 미실행 시 JSON 응답 없이 연결에 실패할 수 있습니다. GET 등 지원하지 않는 메서드는 기본 HTTP 서버의 501 HTML 응답입니다. 백엔드는 통신 오류나 추론 실패를 entities: []로 취급하면 안 됩니다.

## 학습과 검증

```powershell
.\.venv\Scripts\python.exe predict.py
.\.venv\Scripts\python.exe validate.py
```

- predict.py: 여러 점검 문장의 토큰별 라벨 출력
- validate.py: validation.json의 정답과 정확한 위치·종류 기준으로 비교
- validation_result.json: 전체 정밀도·재현율·F1
- validation_details.json: 문장별 정답·예측·오탐·놓침

검증 출력에는 문장과 값이 포함되므로 로컬 생성 결과는 Git에서 제외합니다. 현재 validation.json과 train.py에는 대화 중 만든 가상 예제가 들어 있습니다. 검증 목록은 개발 중 개선에 사용했으므로 독립 최종 테스트 성능으로 보고하면 안 됩니다. 다양한 실제 입력의 정확도를 보장하지 않습니다.

train.py는 models/Eveil-AI를 불러와 추가 학습하고 같은 폴더에 저장합니다. 실행하면 가중치를 덮어쓰므로 의도한 경우에만 실행하세요. 가중치는 이어받지만 optimizer 상태를 재개하는 기능은 없습니다. 모델을 변경하면 검증 후 서버를 재시작합니다.

```powershell
.\.venv\Scripts\python.exe train.py
```

check_model.py와 download_model.py는 초기 실습 코드입니다. pii.py는 이전 독립 튜토리얼이며 현재 HTTP 실행 경로에서는 사용하지 않습니다.

## 로그와 운영 범위

server.py는 요청 본문·원문·탐지 값을 로그나 파일에 저장하지 않으며 기본 요청 로그도 비활성화합니다. validate.py와 predict.py는 로컬 점검을 위해 원문을 출력합니다. HTTP 서버는 루프백 전용 시연용으로 인증·HTTPS·외부 배포 설정이 없습니다. 다른 장비에서 호출하려면 팀 내 접근 제한과 암호화 통신을 별도로 구성해야 합니다. 응답 시간과 부하 한도는 실행 장비에서 측정해야 합니다.
