"""Local, beginner-sized NER fine-tuning exercise. Run from this folder.
Python 3.10/3.11. Install torch separately, then requirements.txt.
Demo data checks wiring only; it is NOT a useful production dataset.
"""
import argparse
import json
import random
import re
from pathlib import Path

TYPES = ['NAME', 'PHONE', 'ADDRESS', 'ACCOUNT', 'CARD', 'RRN', 'EMAIL']
LABELS = ['O'] + [f'{p}-{t}' for t in TYPES for p in ['B', 'I']]
LABEL2ID = {s: i for i, s in enumerate(LABELS)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def demo():
    # Tags are used only to create sample labels. Real model inputs have no tags.
    groups = {
        'train': [
            '이름: <NAME>김민수</NAME>', '담당자는 <NAME>이영희</NAME> 입니다.',
            '연락처: <PHONE>010-1234-5678</PHONE>',
            '전화번호 뒷자리: <PHONE>5678</PHONE>',
            '주소: <ADDRESS>역삼동</ADDRESS>',
            '배송지: <ADDRESS>강남구 햇살아파트 101동 202호</ADDRESS>',
            '계좌번호: <ACCOUNT>123-456-789</ACCOUNT>',
            '카드번호: <CARD>1234-5678-9012-3456</CARD>',
            '주민등록번호 앞자리: <RRN>900101</RRN>',
            '이메일: <EMAIL>sora@example.com</EMAIL>',
            '이메일: <EMAIL>so***@example.com</EMAIL>',
            '내일 회의 내용을 정리해 주세요.', '이름과 전화번호를 입력해 주세요.',
            '주문번호: 123456789012', '서울에 살아요.',
        ],
        'validation': [
            '신청자 이름은 <NAME>박지훈</NAME> 입니다.',
            '휴대폰 번호는 <PHONE>010-****-1111</PHONE> 입니다.',
            '거주 지역은 <ADDRESS>애월읍</ADDRESS> 입니다.',
            '입금 계좌번호는 <ACCOUNT>234-567-890</ACCOUNT> 입니다.',
            '결제 카드번호는 <CARD>2345-6789-0123-4567</CARD> 입니다.',
            '주민등록번호 앞자리는 <RRN>910202</RRN> 입니다.',
            '답장 받을 이메일은 <EMAIL>reply@example.net</EMAIL> 입니다.',
            '주문 목록을 정리해 주세요.',
        ],
        'test': [
            '수령인 이름을 <NAME>최유진</NAME> 으로 적었습니다.',
            '전화번호를 <PHONE>010-2222-3333</PHONE> 으로 남겼습니다.',
            '주소를 <ADDRESS>해운대구</ADDRESS> 로 적었습니다.',
            '계좌번호를 <ACCOUNT>345-678-901</ACCOUNT> 로 보냈습니다.',
            '카드번호를 <CARD>3456-7890-1234-5678</CARD> 로 적었습니다.',
            '주민등록번호 앞자리를 <RRN>920303</RRN> 으로 적었습니다.',
            '이메일을 <EMAIL>contact@example.org</EMAIL> 로 변경했습니다.',
            '부산 여행 계획을 작성해 주세요.',
        ],
    }
    folder = Path('data')
    folder.mkdir(exist_ok=True)
    for split, examples in groups.items():
        path = folder / f'{split}.jsonl'
        if path.exists():
            raise FileExistsError(f'{path} already exists; demo will not overwrite it.')
        rows = []
        for marked in examples:
            parts, entities, pos, cursor = [], [], 0, 0
            for m in re.finditer(r'<([A-Z]+)>(.*?)</\1>', marked):
                before = marked[cursor:m.start()]
                parts.extend([before, m[2]])
                pos += len(before)
                entities.append({'start': pos, 'end': pos + len(m[2]), 'type': m[1]})
                pos += len(m[2])
                cursor = m.end()
            parts.append(marked[cursor:])
            rows.append({'text': ''.join(parts), 'entities': entities})
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        print(f'{path}: {len(rows)} examples')


def read_rows(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text, entities = row['text'], row['entities']
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'{path}:{line_no}: empty/invalid text')
        previous_end = 0
        for entity in sorted(entities, key=lambda x: x['start']):
            s, e, kind = entity['start'], entity['end'], entity['type']
            if type(s) is not int or type(e) is not int or not (0 <= s < e <= len(text)):
                raise ValueError(f'{path}:{line_no}: invalid offsets {entity}')
            if kind not in TYPES or s < previous_end or text[s:e] != text[s:e].strip():
                raise ValueError(f'{path}:{line_no}: label/overlap/outer whitespace error {entity}')
            previous_end = e
        rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no examples')
    return rows


def encode(row, tokenizer, max_length):
    # No silent truncation: long inputs must be split deliberately.
    item = tokenizer(row['text'], return_offsets_mapping=True, truncation=False)
    if len(item['input_ids']) > max_length:
        raise ValueError(f'Input has {len(item["input_ids"])} tokens, limit={max_length}. Split it first.')
    offsets = item.pop('offset_mapping')
    labels = [-100 if s == e else 0 for s, e in offsets]
    for ent in row['entities']:
        s, e = ent['start'], ent['end']
        hits = [i for i, (a, b) in enumerate(offsets) if a < b and a < e and s < b]
        # Reject unrepresentable character boundaries instead of teaching wrong spans.
        if not hits or offsets[hits[0]][0] != s or offsets[hits[-1]][1] != e:
            raise ValueError(f'Token boundary does not match entity: {ent}; text={row["text"]!r}')
        for j, i in enumerate(hits):
            a, b = offsets[i]
            if a < s or b > e:
                raise ValueError('Token crosses entity boundary')
            labels[i] = LABEL2ID[f'{"B" if j == 0 else "I"}-{ent["type"]}']
    item['labels'] = labels
    return item, offsets


def decode(ids, offsets):
    entities, active = [], None
    for idx, (s, e) in zip(ids, offsets):
        if s == e:
            continue
        label = LABELS[int(idx)]
        if label == 'O':
            active = None
            continue
        prefix, kind = label.split('-', 1)
        if prefix == 'B' or active is None or active['type'] != kind:
            active = {'start': s, 'end': e, 'type': kind}
            entities.append(active)
        else:
            active['end'] = e
    return entities


def choose_device(torch, cpu=False):
    return torch.device('cuda' if not cpu and torch.cuda.is_available() else 'cpu')


def score(model, tokenizer, rows, device, max_length):
    import torch
    model.eval()
    counts = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in TYPES}
    negative_total = negative_false_positive = 0
    with torch.inference_mode():
        for row in rows:
            item, offsets = encode(row, tokenizer, max_length)
            inputs = {k: torch.tensor([v], device=device) for k, v in item.items() if k != 'labels'}
            ids = model(**inputs).logits.argmax(-1)[0].tolist()
            pred = {(x['start'], x['end'], x['type']) for x in decode(ids, offsets)}
            gold = {(x['start'], x['end'], x['type']) for x in row['entities']}
            for kind in TYPES:
                p = {x for x in pred if x[2] == kind}
                g = {x for x in gold if x[2] == kind}
                counts[kind]['tp'] += len(p & g)
                counts[kind]['fp'] += len(p - g)
                counts[kind]['fn'] += len(g - p)
            if not gold:
                negative_total += 1
                negative_false_positive += bool(pred)
    def metrics(c):
        tp, fp, fn = c['tp'], c['fp'], c['fn']
        return {**c, 'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
                'f1': 2 * tp / max(2 * tp + fp + fn, 1)}
    total = {k: sum(c[k] for c in counts.values()) for k in ['tp', 'fp', 'fn']}
    return {'micro': metrics(total), 'per_type': {t: metrics(c) for t, c in counts.items()},
            'negative_examples': negative_total, 'negative_examples_with_false_positive': negative_false_positive}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['demo', 'download', 'check', 'train', 'evaluate', 'predict'])
    parser.add_argument('--base', default='models/base')
    parser.add_argument('--out', default='models/pii')
    parser.add_argument('--data', default='data/test.jsonl')
    parser.add_argument('--text')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()
    if args.command == 'demo':
        demo()
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
    if args.command == 'download':
        from transformers import AutoModel
        tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base', use_fast=True)
        base = AutoModel.from_pretrained('klue/roberta-base', use_safetensors=True)
        base.save_pretrained(args.base, safe_serialization=True)
        tokenizer.save_pretrained(args.base)
        print(f'Base model and tokenizer saved to {args.base}')
        return
    torch.manual_seed(42)
    random.seed(42)
    device = choose_device(torch, args.cpu)
    print(f'Device: {device}')
    if args.command in ['train', 'check']:
        tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True, local_files_only=True)
        rows = {s: read_rows(f'data/{s}.jsonl') for s in ['train', 'validation', 'test']}
        seen = set()
        for split, items in rows.items():
            texts = {r['text'] for r in items}
            if seen & texts:
                raise ValueError('Exact text duplicated across splits')
            seen.update(texts)
            for row in items:
                encode(row, tokenizer, args.max_length)
            print(f'{split}: {len(items)} examples checked')
        if args.command == 'check':
            return
        if args.epochs < 1 or args.batch < 1:
            raise ValueError('epochs and batch must be positive')
        out = Path(args.out)
        if out.exists():
            raise FileExistsError(f'{out} exists; use --out models/pii-run2 for a new run')
        model = AutoModelForTokenClassification.from_pretrained(
            args.base, local_files_only=True, num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
        ).to(device)
        encoded = [encode(r, tokenizer, args.max_length)[0] for r in rows['train']]
        loader = torch.utils.data.DataLoader(encoded, batch_size=args.batch, shuffle=True, num_workers=0,
                    collate_fn=DataCollatorForTokenClassification(tokenizer=tokenizer))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        best_f1 = -1.0
        out.mkdir(parents=True)
        write_json(out / 'run_config.json', vars(args))
        for epoch in range(args.epochs):
            model.train()
            loss_total = 0.0
            for step, batch in enumerate(loader, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_total += loss.item()
                print(f'epoch={epoch + 1} step={step}/{len(loader)} loss={loss.item():.4f}', flush=True)
            result = score(model, tokenizer, rows['validation'], device, args.max_length)
            f1 = result['micro']['f1']
            print(f'epoch={epoch + 1} mean_loss={loss_total / len(loader):.4f} validation_exact_f1={f1:.4f}')
            if f1 > best_f1:
                best_f1 = f1
                model.save_pretrained(out, safe_serialization=True)
                tokenizer.save_pretrained(out)
                write_json(out / 'validation_metrics.json', {'epoch': epoch + 1, **result})
        print(f'Best validation model saved: {out}. Run evaluate separately on held-out test data.')
        return
    tokenizer = AutoTokenizer.from_pretrained(args.out, use_fast=True, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(args.out, local_files_only=True).to(device)
    if args.command == 'evaluate':
        result = score(model, tokenizer, read_rows(args.data), device, args.max_length)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args.text:
        parser.error('predict requires --text')
    row = {'text': args.text, 'entities': []}
    item, offsets = encode(row, tokenizer, args.max_length)
    inputs = {k: torch.tensor([v], device=device) for k, v in item.items() if k != 'labels'}
    model.eval()
    with torch.inference_mode():
        ids = model(**inputs).logits.argmax(-1)[0].tolist()
    print(json.dumps({'entities': decode(ids, offsets)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
