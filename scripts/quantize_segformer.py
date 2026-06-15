#!/usr/bin/env python3
import os
import cv2
import numpy as np
import openvino as ov
import nncf
import rospkg

def main():
    # 1. ROS 패키지 및 모델 경로 자동 설정
    rospack = rospkg.RosPack()
    try:
        pkg_path = rospack.get_path('my_robot_segmentation')
    except Exception:
        print("❌ 'my_robot_segmentation' 패키지를 찾을 수 없습니다. ROS 환경을 먼저 source 해주세요.")
        return

    model_dir = os.path.join(pkg_path, 'weights/segformer_ov')
    model_path = os.path.join(model_dir, 'segformer.xml')
    output_path = os.path.join(model_dir, 'segformer_int8.xml')

    print(f"📦 원본 FP32 모델 로드 중: {model_path}")
    core = ov.Core()
    model = core.read_model(model_path)

    # 2. 전처리 파라미터 (기존 추론 엔진과 100% 동일한 수학적 연산)
    input_size = (512, 512)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # 3. 캘리브레이션 데이터 폴더 확인 및 생성
    calib_data_dir = os.path.join(pkg_path, 'data/calibration')
    if not os.path.exists(calib_data_dir):
        os.makedirs(calib_data_dir)
        print(f"📁 캘리브레이션 폴더를 생성했습니다: {calib_data_dir}")
        print("⚠️ 위 폴더에 실제 로봇 주행 이미지(*.jpg 또는 *.png)를 50~200장 정도 넣고 다시 실행해주세요!")
        return

    # 이미지 파일 목록 수집
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [os.path.join(calib_data_dir, f) for f in os.listdir(calib_data_dir) 
                   if f.lower().endswith(supported_extensions)]

    if len(image_files) == 0:
        print(f"❌ {calib_data_dir} 폴더에 이미지가 없습니다.")
        print("💡 바닥 인식 정확도를 유지하려면 실제 주행 환경의 스크린샷 이미지들이 필요합니다.")
        return

    print(f"📸 총 {len(image_files)}장의 이미지를 기반으로 데이터 오차 보정(Calibration)을 준비합니다.")

    # 4. 양자화용 전처리 변환 함수 정의
    def transform_fn(image_path):
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros((1, 3, 512, 512), dtype=np.float32)
        
        # BGR에서 RGB로 변환 (기존 전처리와 동기화)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img_float /= 255.0
        img_float -= mean
        img_float /= std
        
        input_tensor = np.expand_dims(img_float.transpose(2, 0, 1), axis=0)
        return input_tensor

    # 5. NNCF 데이터셋 생성 및 양자화 수행
    calibration_dataset = nncf.Dataset(image_files, transform_fn)

    print("⏱️ INT8 양자화 최적화를 시작합니다. (컴퓨터 사양에 따라 수 분이 소요됩니다)...")
    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        subset_size=min(300, len(image_files)),
        preset=nncf.QuantizationPreset.PERFORMANCE # 속도 최적화 모드
    )

    # 6. 최적화된 INT8 모델 저장
    ov.save_model(quantized_model, output_path)
    print(f"🎉 INT8 양자화 모델 생성 성공!")
    print(f"💾 결과물 저장 완료: {output_path}")

if __name__ == '__main__':
    main()