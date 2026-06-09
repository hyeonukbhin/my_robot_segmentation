#!/usr/bin/env python3
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def main():
    # 1. 저장할 로컬 절대 경로 설정
    save_path = os.path.expanduser('~/ros2_ws/src/my_robot_segmentation/weights/segformer_b0')

    # 2. 가중치를 저장할 폴더가 없다면 자동으로 생성
    os.makedirs(save_path, exist_ok=True)

    print("=========================================")
    print("📥 Segformer 오프라인 가중치 다운로드 시작...")
    print("네트워크 상태에 따라 1~2분 정도 소요될 수 있습니다.")
    print("=========================================")

    # 3. 허깅페이스 서버에서 모델과 프로세서 로드
    processor = SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')

    # 4. 로컬 경로에 영구 저장
    processor.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f"\n✅ 로컬 저장 완료!")
    print(f"📂 저장 위치: {save_path}")

if __name__ == "__main__":
    main()
