from transformers import AutoTokenizer

model_name = "klue/roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("토크나이저 준비 완료")

from transformers import AutoModel

model = AutoModel.from_pretrained(
    model_name,
    use_safetensors=True,
)

print("모델준비완료")

save_path = "./models/base"

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("저장 완료")