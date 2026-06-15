#!/usr/bin/env python3

import os
import cv2
import numpy as np
import openvino as ov
import rospkg 

class SegformerEngineOV:
    """Mode 1 전용: 불필요 레이블 선행 제거 및 원본 해상도 복구 엔진 (FP32/INT8 딕셔너리 선택)"""
    def __init__(self):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('my_robot_segmentation')
        
        # 모델 구조적 한계치 (절대 변경 불가)
        self.input_size = (512, 512)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        self.core = ov.Core()
        
        # ==========================================================
        # 🚀 사용자 설정: 여기서 "int8" 또는 "fp32"로 이름만 바꿔주세요!
        # ==========================================================
        target_mode = "int8" 
        
        # 모델 파일명 매핑 딕셔너리
        model_dict = {
            "fp32": "segformer.xml",
            "int8": "segformer_int8.xml"
        }
        
        # 딕셔너리에서 파일명을 가져옵니다. (오타 방지용 기본값은 fp32)
        model_name = model_dict.get(target_mode, "segformer.xml")
        model_path = os.path.join(pkg_path, f'weights/segformer_ov/{model_name}')
        
        print(f"🔄 OpenVINO 모델 로딩 중: [{target_mode.upper()} 버전] + CPU 극한 최적화 모드...")
        
        # 하드웨어 가속 대신 CPU의 모든 물리 코어를 쥐어짜는 최적화 설정
        config = {
            "PERFORMANCE_HINT": "LATENCY",  # 단일 프레임 지연 시간(Latency) 최소화 집중
            "INFERENCE_NUM_THREADS": "0"    # 가용 물리 코어 자동 최대 할당
        }
        
        # CPU 고정 및 최적화 config 주입
        self.compiled_model = self.core.compile_model(
            model=model_path, 
            device_name="CPU", 
            config=config
        )
        
        self.output_layer = self.compiled_model.output(0)
        print(f"✅ OpenVINO Segformer 엔진 준비 완료! [로드된 파일: {model_name}]")

    def infer(self, cv_rgb):
        original_h, original_w = cv_rgb.shape[:2]

        # 1. 모델 규격에 맞춘 전처리
        img_float = cv2.resize(cv_rgb, self.input_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img_float /= 255.0
        img_float -= self.mean
        img_float /= self.std
        
        input_tensor = np.expand_dims(img_float.transpose(2, 0, 1), axis=0)

        # 2. AI 추론 실행
        results = self.compiled_model([input_tensor])[self.output_layer]
        logits = results[0] 

        # 3. 레이블 압축 및 최적화 추출 (150개 레이블 -> 2개 레이블)
        floor_logit = logits[3, :, :]
        bg_logit = np.max(np.delete(logits, 3, axis=0), axis=0)

        # 4. 원본 해상도 고품질 복구 작업
        floor_logit_up = cv2.resize(floor_logit, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        bg_logit_up = cv2.resize(bg_logit, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # 원본 해상도 스케일에서 두 확률을 비교하여 최종 마스크 생성 (계단 현상 원천 차단)
        floor_mask_smooth = (floor_logit_up > bg_logit_up).astype(np.uint8) * 255
        
        return floor_mask_smooth