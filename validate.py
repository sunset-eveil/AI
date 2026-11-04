import json
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

def merge_predictions(predicted_ids, offsets):
    entities = []
    current_entity = None

    for label_id, (start, end) in zip(predicted_ids, offsets):
       if start == end:
           continue

       label = model.config.id2label[label_id]

       if label == "O":
           current_entity = None
           continue

       prefix, entity_type = label.split("-", 1)

       if prefix == "B":
           current_entity = {
               "start": start,
               "end": end,
               "type": entity_type,
           }
           entities.append(current_entity)

       elif(
            prefix == "I"
            and current_entity is not None
            and current_entity["type"] == entity_type
        ):
           current_entity["end"] = end
       else :
           current_entity = None
    return entities


with open("validation.json", "r", encoding="utf-8") as file:
    validation_examples = json.load(file)

print(len(validation_examples))

total_correct = 0
total_false_positives = 0
total_missed = 0

validation_details = []

for example in validation_examples:

    text = example["text"]
    print("\n", text)

    if not example["entities"]:
        print("개인정보 없음")

    for entity in example["entities"]:
        value = text[entity["start"]:entity["end"]]
        print(value, entity["type"])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offsets = inputs.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    predicted_entities = merge_predictions(predicted_ids, offsets)

    gold_set = {
        (entity["start"], entity["end"], entity["type"])
        for entity in example["entities"]
    }

    predicted_set = {
        (entity["start"], entity["end"], entity["type"])
        for entity in predicted_entities
    }

    correct = gold_set & predicted_set
    false_positives = predicted_set - gold_set
    missed = gold_set - predicted_set

    total_correct += len(correct)
    total_false_positives += len(false_positives)
    total_missed += len(missed)

    def describe_entities(entity_set):
        return[
            {
                "start": start,
                "end": end,
                "type": entity_type,
                "text": text[start:end],
            }
            for start, end, entity_type in sorted(entity_set)
        ]
    validation_details.append({
        "text": text,
        "gold_entities": example["entities"],
        "predicted_entities": predicted_entities,
        "false_positives": describe_entities(false_positives),
        "missed": describe_entities(missed),
    })

    print("맞힌 개인정보 수:", len(correct))
    print("잘못 탐지한 수:", len(false_positives))
    print("놓친 개인정보 수:", len(missed))

    print("합친 예측 결과:", predicted_entities)

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].tolist()
    )

    print("토큰별 원문 위치:", offsets)
    print("모델 예측:")

    for token, label_id in zip(tokens, predicted_ids):
        if token in tokenizer.all_special_tokens:
            continue

        label_name = model.config.id2label[label_id]
        print(token, "->", label_name)

predicted_count = total_correct + total_false_positives
precision = total_correct / predicted_count if predicted_count > 0 else 0.0

gold_count = total_correct + total_missed
recall = total_correct / gold_count if gold_count > 0 else 0.0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0.0
)

print("\n전체 검증 결과")
print("맞힘:", total_correct)
print("오탐:", total_false_positives)
print("놓침:", total_missed)
print(f"정밀도: {precision:.2%}")
print(f"재현율: {recall:.2%}")
print(f"F1: {f1:.2%}")

validation_result = {
    "example_count": len(validation_examples),
    "correct": total_correct,
    "false_positives": total_false_positives,
    "missed": total_missed,
    "precision": precision,
    "recall": recall,
    "f1": f1,
}

with open("validation_result.json", "w", encoding="utf-8") as file:
    json.dump(
        validation_result,
        file,
        ensure_ascii=False,
        indent=2,
    )

with open("validation_details.json", "w", encoding="utf-8") as file:
    json.dump(
        validation_details,
        file,
        ensure_ascii=False,
        indent=2,
    )