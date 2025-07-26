import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
from PIL import Image
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore", message=".*use_fast.*")

# ✅ 1. 모델 로드
model_dir = "../medgemma-local"
processor = AutoProcessor.from_pretrained(model_dir, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_dir,
    torch_dtype=torch.float32,  # CPU에서도 안정적
    device_map="auto" if torch.cuda.is_available() else None
)

# ✅ 2. vqa-rad test 데이터셋 로드
dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")

# ✅ 3. 후처리 함수 (모델 답변 정제)
def clean_answer(raw_answer: str) -> str:
    """모델 출력에서 프롬프트(system/user 텍스트) 제거"""
    text = raw_answer.lower()
    # "model" 이후 부분만 추출 (모델 답변 구간)
    if "model" in text:
        text = text.split("model")[-1]
    return text.strip()

def is_correct_answer(gt: str, pred: str) -> bool:
    """단순 문자열 비교 대신 키워드 포함 여부 기반 정확도 계산"""
    if gt in ["yes", "no"]:
        return gt in pred  # yes/no 포함되면 정답 처리
    return gt in pred     # open-ended는 키워드 포함되면 정답

results = []
num_samples = 5  # 테스트 샘플 수

# ✅ 4. 루프 실행
for i, sample in enumerate(dataset.select(range(num_samples))):
    question = sample["question"]
    answer_gt = sample["answer"].strip().lower()
    image = sample["image"].convert("RGB")

    # ✅ ChatTemplate 기반 메시지 생성
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image", "image": image}
        ]}
    ]

    # ✅ Processor에서 자동으로 이미지 토큰 삽입 (return_dict=True 필수)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    # ✅ 추론
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    # ✅ 답변 정제 + 정확도 계산
    answer_pred_raw = processor.decode(outputs[0], skip_special_tokens=True)
    answer_pred = clean_answer(answer_pred_raw)
    correct = is_correct_answer(answer_gt, answer_pred)

    results.append({
        "index": i,
        "question": question,
        "gt_answer": answer_gt,
        "pred_answer": answer_pred,
        "correct": correct
    })

    print(f"[{i}] Q: {question}\nGT: {answer_gt} | PRED(clean): {answer_pred} | Correct: {correct}\n")

# ✅ 5. 결과를 DataFrame → Excel 저장
df = pd.DataFrame(results)
excel_path = "vqa_rad_test_10.xlsx"
df.to_excel(excel_path, index=False)
print(f"✅ {num_samples}개 샘플 테스트 결과 저장 완료: {os.path.abspath(excel_path)}")

# ✅ 6. 정확도 출력
accuracy = df["correct"].mean() * 100
print(f"✅ Accuracy ({num_samples} samples): {accuracy:.2f}%")
