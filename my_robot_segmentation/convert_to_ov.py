import openvino as ov
import torch
from transformers import SegformerForSemanticSegmentation

# 1. 로컬에 있는 원본 PyTorch 모델 로드
local_path = '/home/bhin/ros2_ws/src/my_robot_segmentation/weights/segformer_b0'
print("로컬 PyTorch 모델을 불러오는 중...")
model = SegformerForSemanticSegmentation.from_pretrained(local_path, local_files_only=True)
model.eval()

# 2. 더미 입력 데이터 생성 (해상도 640x480 기준)
dummy_input = torch.randn(1, 3, 480, 640)

# 3. OpenVINO 모델로 변환
print("🔄 OpenVINO 포맷으로 변환 중... (시간이 조금 걸릴 수 있습니다)")
ov_model = ov.convert_model(model, example_input=dummy_input)

# 4. 변환된 모델 저장
save_path = '/home/bhin/ros2_ws/src/my_robot_segmentation/weights/segformer_ov/segformer.xml'
ov.save_model(ov_model, save_path)
print(f"✅ 변환 완료! 파일이 저장되었습니다: {save_path}")