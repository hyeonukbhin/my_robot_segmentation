import cv2
import numpy as np
import openvino as ov
from transformers import SegformerImageProcessor

class SegformerEngineOV:
    """Mode 1 전용 (초경량화): OpenVINO 가속 기반 Segformer 엔진 (오직 Floor만 추출)"""
    def __init__(self):
        # 불필요한 target_objects 파라미터 제거
        local_path = '/home/bhin/ros2_ws/src/my_robot_segmentation/weights/segformer_b0'
        self.processor = SegformerImageProcessor.from_pretrained(local_path, local_files_only=True)
        
        self.core = ov.Core()
        model_path = '/home/bhin/ros2_ws/src/my_robot_segmentation/weights/segformer_ov/segformer.xml'
        
        print("🔄 OpenVINO 모델 로딩 중 (NUC iGPU 가속 AUTO 모드 - 초경량화)...")
        model = self.core.read_model(model=model_path)
        self.compiled_model = self.core.compile_model(model=model, device_name="AUTO")
        self.output_layer = self.compiled_model.output(0)
        print("✅ OpenVINO Segformer 엔진 준비 완료! (Floor Only Mode)")

    def infer(self, cv_rgb):
        # 1. 전처리
        inputs = self.processor(images=cv_rgb, return_tensors="np")
        pixel_values = inputs["pixel_values"]
        
        # 2. OpenVINO 추론 실행
        results = self.compiled_model([pixel_values])[self.output_layer]
        
        # 3. 후처리
        logits = np.transpose(results[0], (1, 2, 0)) 
        upsampled_logits = cv2.resize(logits, (cv_rgb.shape[1], cv_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        predicted_map = np.argmax(upsampled_logits, axis=-1).astype(np.uint8)

        # 🌟 연산량 최적화: 바닥(클래스 3)만 추출하고 나머지 연산은 전부 생략
        floor_mask = (predicted_map == 3).astype(np.uint8) * 255
        
        # 오직 floor_mask 하나만 반환합니다.
        return floor_mask