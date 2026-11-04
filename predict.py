import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_path = "./models/Eveil-AI"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

model = AutoModelForTokenClassification.from_pretrained(
    model_path,
    local_files_only=True,
)

model.eval()

test_texts = [
    "😀 담당자: 한서윤\n연락처는 031-246-8135입니다.",
    "지후에게 전달하고, 처리 후 지후에게 알려 주세요.",
    "받을 곳은 경기 수원시 영통구 매탄동 101동 202호예요.",
    "연락처 뒷자리는 4826이고, 주문번호는 4826입니다.",
    "입금은 135-802-479로 해 주세요. 접수번호는 135-802-480입니다.",
    "결제 카드 번호는 0000 2222 5555 7777, 유효기간은 08/29예요.",
    "주민등록번호 앞 여섯 자리는 010203이고, 예약 코드는 010204예요.",
    "답장은 qa.team+demo@example.net으로 보내 주세요.",
]

for text in test_texts:
    print("\n", text)

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_ids = outputs.logits.argmax(dim=-1)[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].tolist()
    )

    for token, label_id in zip(tokens, predicted_ids):
        if token in tokenizer.all_special_tokens:
            continue

        label_name = model.config.id2label[label_id]
        print(token, "->", label_name)