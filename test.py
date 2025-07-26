import os
import re
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from PIL import Image
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message=".*use_fast.*")

# ✅ 1) 모델 및 Processor 로드
base_model_dir = "../medgemma-local"          # 사전 학습 모델 경로
finetuned_model_dir = "../medgemma-finetuned"  # Fine-tuned 모델 저장 경로

processor = AutoProcessor.from_pretrained(base_model_dir, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    base_model_dir,
    torch_dtype=torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

print("✅ Model and processor loaded")

# ✅ 2) VQA-RAD 데이터셋 로드 및 answer_type 추가
dataset = load_dataset("flaviagiammarino/vqa-rad")

def add_answer_type(example):
    ans = example["answer"].strip().lower()
    if ans in ["yes", "no"]:
        example["answer_type"] = "yes/no"
    else:
        example["answer_type"] = "open-ended"
    example["formatted_answer"] = f"{example['answer_type']}: {ans}"
    return example

dataset = dataset.map(add_answer_type)
print("✅ Dataset processed with answer_type")

# ✅ 3) Fine-tuning 전처리 함수 (apply_chat_template 사용)
def preprocess(example):
    image = example["image"].convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
        {"role": "user", "content": [
            {"type": "text", "text": example["question"]},
            {"type": "image", "image": image}
        ]}
    ]

    # apply_chat_template으로 image_token 자동 처리
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )
    labels = processor.tokenizer(example["formatted_answer"], return_tensors="pt").input_ids
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    inputs["labels"] = labels.squeeze(0)
    return inputs

# ✅ 4) Arrow 저장 문제 해결 (set_transform 사용)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

train_dataset.set_transform(preprocess)
eval_dataset.set_transform(preprocess)
print("✅ Dataset transform ready for training")

# ✅ 5) Fine-tuning 설정
training_args = TrainingArguments(
    output_dir=finetuned_model_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,  # 테스트용 (실제는 3~5 epoch 권장)
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer
)

# ✅ 6) Fine-tuning 수행
trainer.train()
trainer.save_model(finetuned_model_dir)
processor.save_pretrained(finetuned_model_dir)
print(f"✅ Fine-tuning 완료 및 저장: {finetuned_model_dir}")

# ✅ 7) Fine-tuned 모델 로드 후 평가
processor = AutoProcessor.from_pretrained(finetuned_model_dir)
model = AutoModelForImageTextToText.from_pretrained(
    finetuned_model_dir,
    torch_dtype=torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

def parse_answer(raw_text: str):
    raw_text = raw_text.lower().strip()
    match = re.match(r"(yes/no|open-ended)\s*:\s*(.*)", raw_text)
    if match:
        return match.group(1), match.group(2).strip()
    return "unknown", raw_text

def is_correct_answer(gt: str, pred: str, answer_type: str) -> bool:
    if answer_type == "yes/no":
        return gt in ["yes", "no"] and gt in pred
    return gt in pred

# ✅ 8) 테스트 및 결과 저장
results = []
num_samples = 10

for i, sample in enumerate(dataset["test"].select(range(num_samples))):
    question = sample["question"]
    gt = sample["answer"].strip().lower()
    image = sample["image"].convert("RGB")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image", "image": image}
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    pred_raw = processor.decode(outputs[0], skip_special_tokens=True)
    answer_type, pred_answer = parse_answer(pred_raw)
    correct = is_correct_answer(gt, pred_answer, answer_type)

    results.append({
        "index": i,
        "question": question,
        "gt_answer": gt,
        "answer_type": answer_type,
        "pred_answer": pred_answer,
        "correct": correct
    })

    print(f"[{i}] Q: {question}\nGT: {gt} | TYPE: {answer_type} | PRED: {pred_answer} | Correct: {correct}\n")

df = pd.DataFrame(results)
excel_path = "vqa_rad_finetuned_results.xlsx"
df.to_excel(excel_path, index=False)
print(f"✅ {num_samples}개 샘플 결과 저장 완료: {os.path.abspath(excel_path)}")

accuracy = df["correct"].mean() * 100
print(f"✅ Accuracy ({num_samples} samples): {accuracy:.2f}%")
