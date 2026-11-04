from transformers import AutoTokenizer, AutoModel

model_path = "./models/base"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

model =AutoModel.from_pretrained(
    model_path,
    local_files_only=True,
)

print("완료")

text = "민수에게 연락해 줘."

inputs = tokenizer(
    text,
    return_offsets_mapping=True,
)

print("토큰:", tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
print("토큰 번호:", inputs["input_ids"])
print("원문 위치:", inputs["offset_mapping"])

entities = [
    {"start": 0, "end" : 2, "type": "NAME"}
]

for entity in entities:
    value = text[entity["start"]:entity["end"]]
    print("정답:", value, entity["type"])