#!/usr/bin/env python3

import os
import cv2
import numpy as np
import openvino as ov
import rospkg 

class SegformerEngineOV:
    """Mode 1 전용: 불필요 레이블 선행 제거 및 원본 해상도 복구 엔진 (CPU 레거시 최적화 버전)"""
    def __init__(self):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('my_robot_segmentation')
        
        # 모델 구조적 한계치 (절대 변경 불가)
        self.input_size = (512, 512)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        self.core = ov.Core()
        
        # [기존 유지] 원본 FP32 모델 경로 그대로 사용
        model_path = os.path.join(pkg_path, 'weights/segformer_ov/segformer.xml')
        
        print("🔄 OpenVINO 모델 로딩 중 (CPU 극한 최적화 & 고해상도 모드)...")
        
        # ==========================================================
        # 🚀 하드웨어 가속 대신 CPU의 모든 물리 코어를 쥐어짜는 최적화 설정
        # ==========================================================
        config = {
            "PERFORMANCE_HINT": "LATENCY",  # 처리량보다 단일 프레임 지연 시간(Latency) 최소화에 집중
            "INFERENCE_NUM_THREADS": "0"    # 0 설정 시 시스템의 가용 물리 코어를 자동으로 최대 할당
        }
        
        # device_name을 "AUTO"에서 "CPU"로 고정하고 최적화 config를 주입합니다.
        self.compiled_model = self.core.compile_model(
            model=model_path, 
            device_name="CPU", 
            config=config
        )
        
        self.output_layer = self.compiled_model.output(0)
        print("✅ OpenVINO Segformer 엔진(CPU_LATENCY 최적화) 준비 완료!")

    def infer(self, cv_rgb):
        original_h, original_w = cv_rgb.shape[:2]

        # 1. 모델 규격에 맞춘 전처리 (기존 무거운 허깅페이스 라이브러리와 100% 동일한 동작)
        img_float = cv2.resize(cv_rgb, self.input_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img_float /= 255.0
        img_float -= self.mean
        img_float /= self.std
        
        input_tensor = np.expand_dims(img_float.transpose(2, 0, 1), axis=0)

        # 2. AI 추론 실행
        results = self.compiled_model([input_tensor])[self.output_layer]
        logits = results[0] 

        # ==========================================================
        # 🚀 기존 레이블 최적화 로직 100% 유지
        # ==========================================================
        # 150개 중 우리가 필요한 '바닥(3번)' 확률맵만 떼어냅니다.
        floor_logit = logits[3, :, :]
        
        # 나머지 149개 레이블은 겹쳐서 가장 높은 '배경 확률맵' 1장으로 압축해 버립니다. (148개 폐기)
        bg_logit = np.max(np.delete(logits, 3, axis=0), axis=0)

        # ==========================================================
        # 🚀 원본 해상도 복구 로직 100% 유지
        # ==========================================================
        # 연산이 150장 -> 2장으로 줄었으므로, 이 2장만 카메라 원본 해상도로 부드럽게 확대합니다.
        floor_logit_up = cv2.resize(floor_logit, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        bg_logit_up = cv2.resize(bg_logit, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # 원본 해상도 스케일에서 두 확률을 비교하여 최종 마스크 생성 (계단 현상 원천 차단)
        floor_mask_smooth = (floor_logit_up > bg_logit_up).astype(np.uint8) * 255
        
        return floor_mask_smooth