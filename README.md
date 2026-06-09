# ROS 1 OpenVINO Floor Segmentation (Lightweight)

이 패키지는 Intel NUC 등 환경에서 CPU 및 iGPU(내장 그래픽)를 활용하여 초경량/초고속으로 주행 가능 영역(Floor)을 세그멘테이션하는 ROS 1 전용 패키지입니다. 

불필요한 3D 투영 연산을 모두 제거하고, OpenVINO 가속을 이용해 `floor_mask`만을 단일 발행하도록 극단적으로 최적화되어 있습니다.

## 1. 환경 요건 (Requirements)
* **OS:** Ubuntu 20.04 (Noetic) 등 ROS 1 환경
* **Python:** Python 3.x
* **Hardware:** Intel CPU / iGPU 권장 (OpenVINO 가속용)

## 2. 사전 설치 (Dependencies)
ROS 1 환경의 터미널(또는 가상환경)에서 아래 파이썬 패키지를 설치해 주세요.
```bash
pip3 install openvino transformers numpy opencv-python rospkg
```

## 3. 입출력 토픽 사양 (Inputs / Outputs)

노드가 활성화되면 대역폭 낭비와 지연(Latency)을 원천 차단하기 위해 단일 큐(`queue_size=1`) 파이프라인으로 통신합니다.

| 분류 | 토픽명 | 메시지 타입 | 사양 및 설명 |
| :--- | :--- | :--- | :--- |
| **Input** | `/d456_camera/color/image_raw` | `sensor_msgs/Image` | 원본 입력 카메라 영상 (포맷: `rgb8` 또는 `bgr8`) |
| **Output** | `/segmentation/floor_mask` | `sensor_msgs/Image` | 추출된 바닥 영역 이진 마스크 (포맷: `mono8`, 바닥: 255 / 배경: 0) |

## 4. 노드 실행
```bash
rosrun my_robot_segmentation segmentation_node.py
```