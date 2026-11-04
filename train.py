from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
import torch
from transformers import set_seed

model_path = "./models/Eveil-AI"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

entity_types = [
    "NAME",
    "PHONE",
    "ADDRESS",
    "ACCOUNT",
    "CARD",
    "RRN",
    "EMAIL",
]

labels = ["O"]

for entity_type in entity_types:
    labels.append(f"B-{entity_type}")
    labels.append(f"I-{entity_type}")


label2id = {}
id2label = {}

for index, label in enumerate(labels):
    label2id[label] = index
    id2label[index] = label

set_seed(42)

model = AutoModelForTokenClassification.from_pretrained(
    model_path,
    local_files_only=True,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
)

train_examples = [
    {
        "text": "민수에게 연락해 줘.",
        "entities": [
            {"start": 0, "end": 2, "type": "NAME"}
        ],
    },
    {
        "text": "회의 내용을 요약해 줘",
        "entities": [],
    },
        {
        "text": "전화번호는 010-1234-5678입니다.",
        "entities": [
            {"start": 6, "end": 19, "type": "PHONE"}
        ],
    },
    {
        "text": "주문번호는 234567890123입니다.",
        "entities": [],
    },
    {
        "text": "전화번호는 01012345678입니다.",
        "entities": [
            {"start": 6, "end": 17, "type": "PHONE"}
        ],
    },
    {
        "text": "담당자는 수진입니다.",
        "entities": [
            {"start": 5, "end": 7, "type": "NAME"}
        ],
    },
    {
        "text": "전화번호는 061-123-4567입니다.",
        "entities": [
            {"start": 6, "end": 18, "type": "PHONE"}
        ],
    },
    {
        "text": "연락처: 0612345678",
        "entities": [
            {"start": 5, "end": 15, "type": "PHONE"}
        ],
    },
    {
        "text": "대표번호: 02-1234-5678",
        "entities": [
            {"start": 6, "end": 18, "type": "PHONE"}
        ],
    },
    {
        "text": "전 &화 번(ㄱ*ㄴ)호는 01012345678입니다.",
        "entities": [
            {"start": 14, "end": 25, "type": "PHONE"}
        ],
    },
    {
        "text": "전 &화 번(ㄷ*ㄹ)호를 입력해 주세요.",
        "entities": [],
    },
]

new_examples = [
    ("배송지는 부산 해운대구입니다.", "부산 해운대구", "ADDRESS"),
    ("입금 계좌번호: 123-456-789", "123-456-789", "ACCOUNT"),
    ("결제 카드번호: 0000-1111-2222-3333", "0000-1111-2222-3333", "CARD"),
    ("주민등록번호 앞자리: 990231", "990231", "RRN"),
    ("회신은 sora@example.com으로 보내 주세요.", "sora@example.com", "EMAIL"),
        # 이름이 문장 앞이나 중간에 나오고, 조사가 붙는 경우
    ("하린에게 자료를 보내 주세요.", "하린", "NAME"),
    ("이번 담당자는 김도윤입니다.", "김도윤", "NAME"),
    ("파일은 서준님께 전달해 줘.", "서준", "NAME"),

    # 공백이 있어도 하나의 주소 범위로 학습
    ("받을 곳은 서울 마포구입니다.", "서울 마포구", "ADDRESS"),
    ("주소: 인천 연수구 송도동", "인천 연수구 송도동", "ADDRESS"),

    # 숫자 모양이 비슷해도 앞뒤 문맥에 따라 종류가 다름
    ("입금할 계좌: 246-802-135", "246-802-135", "ACCOUNT"),
    ("765432109876은 정산 계좌번호예요.", "765432109876", "ACCOUNT"),
    ("카드 번호를 0000-2222-4444-6666으로 적어 주세요.", "0000-2222-4444-6666", "CARD"),
    ("결제 카드: 0000111122224444", "0000111122224444", "CARD"),

    # 앞자리 6자리를 하나의 범위로 학습
    ("주민번호 앞 6자리는 000000입니다.", "000000", "RRN"),
    ("주민등록번호 앞부분만 적습니다: 000101", "000101", "RRN"),

    # 이메일 시작 위치와 표현을 다양하게
    ("메일 주소: help@example.net", "help@example.net", "EMAIL"),
    ("notice@example.com으로 회신해 줘.", "notice@example.com", "EMAIL"),
    ("회신 주소는 help.desk@example.org 입니다.", "help.desk@example.org", "EMAIL"),

     # 이름 4개: 호칭, 조사, 문장 중간, 성 없는 이름
    ("서아님께 초대장을 보내 주세요.", "서아", "NAME"),
    ("확인 담당은 백하윤 대리입니다.", "백하윤", "NAME"),
    ("검토가 끝나면 다온에게도 알려 줘.", "다온", "NAME"),
    ("이번 안내는 최서진이 맡았어요.", "최서진", "NAME"),

    # 전화번호 4개: 유선, 공백, 일부 가림, 뒷자리
    ("사무실 직통은 042-258-0369예요.", "042-258-0369", "PHONE"),
    ("문자 받을 번호: 010 3579 2468", "010 3579 2468", "PHONE"),
    ("기록에 남은 연락처는 010-****-7302입니다.", "010-****-7302", "PHONE"),
    ("휴대폰 번호 뒷자리는 6824로 적어 주세요.", "6824", "PHONE"),

    # 주소 4개: 도로명, 상세주소, 읍, 면
    ("물건은 인천 부평구 새봄로 27로 보내 주세요.", "인천 부평구 새봄로 27", "ADDRESS"),
    ("수령 장소: 대구 수성구 범어동 별빛아파트 203동 1102호", "대구 수성구 범어동 별빛아파트 203동 1102호", "ADDRESS"),
    ("거주 지역은 봉담읍이에요.", "봉담읍", "ADDRESS"),
    ("방문 지역은 가평군 상면으로 적어 주세요.", "가평군 상면", "ADDRESS"),

    # 계좌번호 4개: 하이픈 유무, 끝자리, 접수번호와 구분
    ("환불받을 계좌는 402-716-835로 등록해 주세요.", "402-716-835", "ACCOUNT"),
    ("정산 계좌 803517264902에 입금해 줘.", "803517264902", "ACCOUNT"),
    ("입금 계좌 끝 네 자리는 7814입니다.", "7814", "ACCOUNT"),
    ("환불 계좌: 308-715-642 / 접수번호: 308-715-643", "308-715-642", "ACCOUNT"),

    # 카드번호 4개: 공백, 일부 가림, 끝자리, 유효기간과 구분
    ("등록할 카드번호는 0000 3333 5555 8888이에요.", "0000 3333 5555 8888", "CARD"),
    ("사용한 카드는 0000-****-****-1111로 표시돼요.", "0000-****-****-1111", "CARD"),
    ("결제한 카드의 끝 네 자리는 5290입니다.", "5290", "CARD"),
    ("카드번호: 0000-4444-5555-9999 / 유효기간: 07/31", "0000-4444-5555-9999", "CARD"),

    # 주민등록번호 3개: 앞자리, 가려진 전체 번호, 가려진 뒷자리
    ("본인 확인용 주민번호 앞자리는 001231입니다.", "001231", "RRN"),
    ("등록된 주민등록번호는 000231-*******로 보입니다.", "000231-*******", "RRN"),
    ("주민등록번호 뒷자리는 0******로 적혀 있어요.", "0******", "RRN"),

    # 이메일 3개: 기호, 괄호, 일부 가림
    ("자료는 ops_team+reply@example.org로 보내 줘.", "ops_team+reply@example.org", "EMAIL"),
    ("회신 주소(help.center@example.net)를 확인해 주세요.", "help.center@example.net", "EMAIL"),
    ("현재 표시된 이메일은 mi***@example.com입니다.", "mi***@example.com", "EMAIL"),

        # 지역번호를 포함한 번호 전체가 PHONE
    ("부재 시 063-274-8190으로 연락해 주세요.", "063-274-8190", "PHONE"),
    ("문의는 055-381-6402로 부탁해요.", "055-381-6402", "PHONE"),
    ("담당자와 통화하려면 041-726-3091로 전화해 줘.", "041-726-3091", "PHONE"),

        # 주민번호와 일반 코드가 함께 있어도 주민번호만 표시
    ("주민번호는 000101-*******로 표시됩니다. 상품코드는 000101입니다.", "000101-*******", "RRN"),

    # 순서가 반대여도 문맥으로 구분
    ("접수코드는 001230이고, 주민등록번호 앞자리는 001231입니다.", "001231", "RRN"),
]

for text, value, entity_type in new_examples:
    start = text.index(value)
    end = start + len(value)

    train_examples.append({
        "text": text,
        "entities": [
            {"start": start, "end": end, "type": entity_type}
        ]
    })

# 숫자가 있어도 개인정보가 아닌 예제
train_examples.extend([
    # 주민번호처럼 보이는 숫자라도 주문코드라면 탐지하지 않음
    {"text": "주문코드는 000215로 입력해 주세요.", "entities": []},
    {"text": "조회할 상품코드: 001130", "entities": []},

    {"text": "상품코드: 246-802-136", "entities": []},
    {"text": "접수번호: 246802135790", "entities": []},
    {"text": "회의 날짜 코드는 000102입니다.", "entities": []},
        # 전화번호처럼 생겼지만 문맥상 주문번호
    {"text": "주문번호는 010-3579-2469입니다.", "entities": []},

    # 항목명만 있고 실제 값이 없음
    {"text": "이름과 이메일, 전화번호를 작성해 주세요.", "entities": []},

    # 시 이름만 나온 경우는 주소로 표시하지 않음
    {"text": "서울에서 열리는 회의는 11월 18일 오후 3시입니다.", "entities": []},

    # 번호 종류를 판단할 단서가 없음
    {"text": "7284", "entities": []},

        {"text": "행사 접수번호는 043-718-2960으로 발급됐어요.", "entities": []},
    {"text": "배송 문의 시 접수 ID 02-413-7826을 알려 주세요.", "entities": []},
    {"text": "출석 확인 코드는 120305입니다.", "entities": []},
    {"text": "신청 내역을 조회하려면 확인 코드 240619를 입력해 주세요.", "entities": []},

    # 같은 숫자라도 두 번째 5728만 전화번호 뒷자리
    {
        "text": "발송 번호는 5728이고, 휴대폰 뒷자리도 5728이에요.",
        "entities": [
            {"start": 24, "end": 28, "type": "PHONE"}
        ],
    },

    # 한 문장에 있는 계좌번호와 카드번호를 각각 표시
    {
        "text": "환불받을 계좌는 7314-209-586이고, 결제 카드번호는 0000-7777-****-3311이에요.",
        "entities": [
            {"start": 9, "end": 21, "type": "ACCOUNT"},
            {"start": 34, "end": 53, "type": "CARD"}
        ],
    },

])

example = train_examples[0]

encoded = tokenizer(
    example["text"],
    return_offsets_mapping=True,
)

tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
offsets = encoded["offset_mapping"]

token_labels = []

for start, end in offsets:
    if start == end:
        token_labels.append(-100)
    else:
        token_labels.append(label2id["O"])

for entity in example["entities"]:
    first_token = True

    for index, (start, end) in enumerate(offsets):
        if start == end:
            continue

        if entity["start"] <= start and end <= entity["end"]:
            if first_token:
                label = f"B-{entity['type']}"
                first_token = False
            else:
                label = f"I-{entity['type']}"

            token_labels[index] = label2id[label]

for token,label_id in zip(tokens, token_labels):
    if label_id == -100:
        label_name = "제외"
    else:
        label_name = id2label[label_id]

def prepare_example(example):
    encoded = tokenizer(
        example["text"],
        return_offsets_mapping=True,
    )

    offsets = encoded.pop("offset_mapping")
    token_labels = []

    for start, end in offsets:
        if start == end:
            token_labels.append(-100)
        else:
            token_labels.append(label2id["O"])

    for entity in example["entities"]:
        matching_tokens = []

        for index, (start, end) in enumerate(offsets):
            if start == end:
                continue

            if start < entity["end"] and entity["start"] < end:
                matching_tokens.append(index)

        if not matching_tokens:
            raise ValueError("정답 범위에 해당하는 토큰이 없습니다.")

        first = matching_tokens[0]
        last = matching_tokens[-1]

        if (
            offsets[first][0] != entity["start"]
            or offsets[last][1] != entity["end"]
        ):
            raise ValueError("정답 경계와 토큰 경계가 맞지 않습니다.")

        for position, index in enumerate(matching_tokens):
            prefix = "B" if position == 0 else "I"

            label = f"{prefix}-{entity['type']}"

            token_labels[index] = label2id[label]

    encoded["labels"] = token_labels

    return encoded

prepared_examples = []

for example in train_examples:
    prepared = prepare_example(example)

    prepared_examples.append(prepared)

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    return_tensors="pt",
)

batch = data_collator(prepared_examples)

# 모델 학습
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.00002
)

for step in range(150):
    model.train()

    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()
    optimizer.step()

    print(f"{step +1}번째 수정 전 손실: {loss.item():.4f}")

save_path = "./models/Eveil-AI"

model.save_pretrained(
    save_path,
    safe_serialization=True,
)

tokenizer.save_pretrained(save_path)

print("완료")
