import os
import sys
import cv2
import torch
import numpy as np
import inspect
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class SegformerEngine:
    """모드 1, 2를 위한 RGB 전용 허깅페이스 엔진"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model.to(self.device)
        self.model.eval()

    def infer(self, cv_rgb, cv_depth=None):
        # 뎁스는 무시하고 RGB만 사용
        inputs = self.processor(images=cv_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=cv_rgb.shape[:2], mode="bilinear", align_corners=False
        )
        predicted_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # ADE20K 데이터셋 기준: 벽(0), 바닥(3) 마스크 추출
        wall_mask = (predicted_map == 0).astype(np.uint8) * 255
        floor_mask = (predicted_map == 3).astype(np.uint8) * 255
        return wall_mask, floor_mask


class NYURGBDEngine:
    """모드 3, 4를 위한 실내 RGB-D ESANet 엔진 (완전 융합 버전)"""
    def __init__(self, weight_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ESANet 절대 경로 강제 주입
        esanet_path = '/home/bhin/ros2_ws/src/my_robot_segmentation/my_robot_segmentation/ESANet'
        if esanet_path not in sys.path:
            sys.path.append(esanet_path)
            
        try:
            from src.models.model import ESANet
        except ImportError as e:
            raise RuntimeError(f"ESANet 로드 실패 (패키지 누락 또는 경로 문제): {e}")

        # 1. Git Checkout된 롤백 코드의 파라미터 구조 자동 추적 및 조립
        sig = inspect.signature(ESANet.__init__)
        accepted_args = sig.parameters.keys()

        # 2. 뼈대 생성 파라미터 딕셔너리 구축
        kwargs = {
            'height': 480,
            'width': 640,
            'num_classes': 40,
            'encoder_rgb': 'resnet34',
            'encoder_depth': 'resnet34',
            'encoder_block': 'NonBottleneck1D',
            'context_module': 'ppm',
            'pretrained_on_imagenet': False
        }

        # 3. 과거 가중치(512채널 레거시 브랜치) 규격에 부합하도록 채널 자동 확장
        if 'channels_decoder' in accepted_args:
            kwargs['channels_decoder'] = [512, 256, 128]
        elif 'decoder_channels' in accepted_args:
            kwargs['decoder_channels'] = [512, 256, 128]

        # 4. 체크아웃한 레거시 버전에 decoder 인자가 살아있을 경우 포함
        if 'decoder' in accepted_args:
            kwargs['decoder'] = 'upsampling'

        print("🔄 ESANet 네트워크를 조립하는 중... (2021 Legacy 완전 안전 매칭)")
        self.model = ESANet(**kwargs)

        # 5. 가중치(.pth) 로드 및 레이어 결합
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        print("✅ ESANet 공식 가중치 로드 성공!")

    def infer(self, cv_rgb, cv_depth):
        # 입력 오리지널 영상 스케일 저장
        orig_h, orig_w = cv_rgb.shape[:2]

        # 1. 뎁스 채널 전처리 (ESANet 표준 해상도 640x480 다운샘플링 및 텐서 배치화)
        cv_depth_resized = cv2.resize(cv_depth, (640, 480))
        depth_tensor = torch.from_numpy(cv_depth_resized).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # 2. RGB 채널 전처리 (리사이즈 및 0~1 정규화)
        cv_rgb_resized = cv2.resize(cv_rgb, (640, 480))
        rgb_tensor = torch.from_numpy(cv_rgb_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # 2-1. ImageNet 도메인 표준 정규화 계수 연산 (추론 정밀도 확보)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb_tensor = ((rgb_tensor - mean) / std).to(self.device)

        # 3. 네트워크 피드포워드 추론 실행
        with torch.no_grad():
            outputs = self.model(rgb_tensor, depth_tensor)
            
        predicted_map = outputs.argmax(dim=1)[0].cpu().numpy()

        # 4. NYUv2 40-Class 표준 매핑 추출 (1: 벽, 2: 바닥)
        wall_mask = (predicted_map == 1).astype(np.uint8) * 255
        floor_mask = (predicted_map == 2).astype(np.uint8) * 255

        # 5. 원본 제어 주기(ROS 2 토픽 규격)에 맞게 픽셀 복원 후 반환
        wall_mask = cv2.resize(wall_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        floor_mask = cv2.resize(floor_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        return wall_mask, floor_mask