from transformers import AutoModelForSemanticSegmentation, AutoConfig

model_name = "nvidia/segformer-b0-finetuned-ade-512-512" # 현재 사용 중인 모델명
config = AutoConfig.from_pretrained(model_name)

# 1. 모델이 알고 있는 클래스 개수 확인
print(f"모델 출력 레이어 클래스 수: {config.num_labels}")

# 2. 클래스 인덱스와 이름 확인
for idx, label in config.id2label.items():
    print(f"인덱스 {idx}: {label}")