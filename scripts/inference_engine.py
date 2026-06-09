#!/usr/bin/env python3

import os
import cv2
import numpy as np
import openvino as ov
import rospkg  # ROS 1 패키지 경로를 찾기 위한 라이브러리 추가
from transformers import SegformerImageProcessor

class SegformerEngineOV:
    """Mode 1 전용 (초경량화): OpenVINO 가속 기반 Segformer 엔진 (오직 Floor만 추출)"""
    def __init__(self):
        # 🌟 경로 동적 탐색 (수정됨)
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('my_robot_segmentation')
        
        local_path = os.path.join(pkg_path, 'weights/segformer_b0')
        self.processor = SegformerImageProcessor.from_pretrained(local_path, local_files_only=True)
        
        self.core = ov.Core()
        model_path = os.path.join(pkg_path, 'weights/segformer_ov/segformer.xml')
        
        print("🔄 OpenVINO 모델 로딩 중 (NUC iGPU 가속 AUTO 모드 - 초경량화)...")
        model = self.core.read_model(model=model_path)
        self.compiled_model = self.core.compile_model(model=model, device_name="AUTO")
        self.output_layer = self.compiled_model.output(0)
        print("✅ OpenVINO Segformer 엔진 준비 완료! (Floor Only Mode)")

    def infer(self, cv_rgb):
        inputs = self.processor(images=cv_rgb, return_tensors="np")
        pixel_values = inputs["pixel_values"]
        
        results = self.compiled_model([pixel_values])[self.output_layer]
        
        logits = np.transpose(results[0], (1, 2, 0)) 
        upsampled_logits = cv2.resize(logits, (cv_rgb.shape[1], cv_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        predicted_map = np.argmax(upsampled_logits, axis=-1).astype(np.uint8)

        floor_mask = (predicted_map == 3).astype(np.uint8) * 255
        
        return floor_mask