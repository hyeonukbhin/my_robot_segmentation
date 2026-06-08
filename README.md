# My Robot Segmentation (Ablation Study)

이 패키지는 자율주행 모바일 로봇의 실내 주행 가능 영역(Floor/Corridor) 인식을 위해, RGB 모델과 RGB-D 모델의 성능을 비교하고 뎁스(Depth) 후처리 필터링의 효과를 검증하는 절제 연구(Ablation Study)용 ROS 2 패키지입니다.

## 📌 Requirements
* ROS 2 (Humble / Foxy)
* Python 3.8+
* PyTorch (CUDA 지원 권장)
* Hugging Face `transformers`

설치 방법:
```bash
pip install -r requirements.txt
sudo apt install ros-<distro>-cv-bridge ros-<distro>-message-filters