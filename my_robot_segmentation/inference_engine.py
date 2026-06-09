import os
import sys
import cv2
import torch
import numpy as np
import inspect
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class SegformerEngine:
    """모드 1, 2를 위한 RGB 전용 허깅페이스 엔진"""
    def __init__(self, target_objects):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 🌟 상위 노드에서 주입받은 객체만 절대적으로 사용 (기본값 삭제)
        self.target_objects = target_objects 
        
        local_path = '/home/bhin/ros2_ws/src/my_robot_segmentation/weights/segformer_b0'
        
        self.processor = SegformerImageProcessor.from_pretrained(local_path, local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(local_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def infer(self, cv_rgb, cv_depth=None):
        inputs = self.processor(images=cv_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=cv_rgb.shape[:2], mode="bilinear", align_corners=False
        )
        predicted_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        wall_mask = (predicted_map == 0).astype(np.uint8) * 255
        floor_mask = (predicted_map == 3).astype(np.uint8) * 255
        ceiling_mask = (predicted_map == 5).astype(np.uint8) * 255
        
        object_mask = np.zeros_like(predicted_map, dtype=np.uint8)
        object_details = {}
        
        # 🌟 등록된 리스트에 없으면 마스크 생성 자체를 건너뜀
        for obj_id, obj_name in self.target_objects.items():
            mask = (predicted_map == obj_id).astype(np.uint8) * 255
            object_details[obj_name] = mask
            object_mask = cv2.bitwise_or(object_mask, mask)
        
        return wall_mask, floor_mask, ceiling_mask, object_mask, object_details

class NYURGBDEngine:
    """모드 3, 4를 위한 실내 RGB-D ESANet 엔진"""
    def __init__(self, weight_path, target_objects):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 🌟 상위 노드에서 주입받은 객체만 절대적으로 사용
        self.target_objects = target_objects 
        
        esanet_path = '/home/bhin/ros2_ws/src/my_robot_segmentation/my_robot_segmentation/ESANet'
        if esanet_path not in sys.path:
            sys.path.append(esanet_path)
            
        try:
            from src.models.model import ESANet
        except ImportError as e:
            raise RuntimeError(f"ESANet 로드 실패: {e}")

        sig = inspect.signature(ESANet.__init__)
        accepted_args = sig.parameters.keys()

        kwargs = {
            'height': 480, 'width': 640, 'num_classes': 40,
            'encoder_rgb': 'resnet34', 'encoder_depth': 'resnet34',
            'encoder_block': 'NonBottleneck1D', 'context_module': 'ppm',
            'pretrained_on_imagenet': False
        }

        if 'channels_decoder' in accepted_args:
            kwargs['channels_decoder'] = [512, 256, 128]
        elif 'decoder_channels' in accepted_args:
            kwargs['decoder_channels'] = [512, 256, 128]

        if 'decoder' in accepted_args:
            kwargs['decoder'] = 'upsampling'

        self.model = ESANet(**kwargs)
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def infer(self, cv_rgb, cv_depth):
        cv_depth_resized = cv2.resize(cv_depth, (640, 480))
        depth_tensor = torch.from_numpy(cv_depth_resized).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        cv_rgb_resized = cv2.resize(cv_rgb, (640, 480))
        rgb_tensor = torch.from_numpy(cv_rgb_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        rgb_tensor = (rgb_tensor - mean) / std

        with torch.no_grad():
            outputs = self.model(rgb_tensor, depth_tensor)
            
        predicted_map = outputs.argmax(dim=1)[0].cpu().numpy()

        wall_mask = (predicted_map == 1).astype(np.uint8) * 255
        floor_mask = (predicted_map == 2).astype(np.uint8) * 255
        ceiling_mask = (predicted_map == 22).astype(np.uint8) * 255

        object_mask = np.zeros_like(predicted_map, dtype=np.uint8)
        object_details = {}
        
        for obj_id, obj_name in self.target_objects.items():
            mask = (predicted_map == obj_id).astype(np.uint8) * 255
            object_details[obj_name] = mask
            object_mask = cv2.bitwise_or(object_mask, mask)

        return wall_mask, floor_mask, ceiling_mask, object_mask, object_details