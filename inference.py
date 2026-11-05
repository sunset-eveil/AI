from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

print("old")

model_path = Path(__file__).resolve().parent / "models" / "Eveil-AI-v2"

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

def detect_entities(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=False,
    )

    if inputs["input_ids"].shape[1] > 256:
        raise ValueError("INPUT_TOO_LONG")

    offsets = inputs.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    return merge_predictions(predicted_ids, offsets)
