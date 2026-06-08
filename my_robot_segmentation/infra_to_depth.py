#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import message_filters
import numpy as np

class InfraToDepthConverter(Node):
    def __init__(self):
        super().__init__('infra_to_depth_converter')
        self.bridge = CvBridge()

        # OpenCV StereoBM 매칭 객체 생성 (파라미터는 환경에 맞게 조절 가능)
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

        # 1. 구독자(Subscriber) 설정 - /camera 네임스페이스 기준
        self.sub_infra1 = message_filters.Subscriber(self, Image, '/d456_camera/infra1/image_rect_raw')
        self.sub_infra2 = message_filters.Subscriber(self, Image, '/d456_camera/infra2/image_rect_raw')
        self.sub_info1 = message_filters.Subscriber(self, CameraInfo, '/d456_camera/infra1/camera_info')

        # 2. Jazzy 환경에서 시간 어긋남을 방지하기 위한 ApproximateTime 동기화 필터 (큐 크기: 10)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_infra1, self.sub_infra2, self.sub_info1], 
            queue_size=10, 
            slop=0.05  # 최대 0.05초 차이까지는 같은 프레임으로 매칭
        )
        self.ts.registerCallback(self.image_callback)

        # 3. 발행자(Publisher) 설정 - 복원된 뎁스 이미지 내보내기
        self.pub_depth = self.create_publisher(Image, '/d456_camera/depth/recovered_image', 10)
        self.get_logger().info('ROS 2 Jazzy Infra-to-Depth 복원 스크립트가 시작되었습니다.')

    def image_callback(self, msg_infra1, msg_infra2, msg_info1):
        try:
            # ROS 2 이미지 메시지를 OpenCV 이미지(흑백)로 변환
            cv_infra1 = self.bridge.imgmsg_to_cv2(msg_infra1, desired_encoding='mono8')
            cv_infra2 = self.bridge.imgmsg_to_cv2(msg_infra2, desired_encoding='mono8')

            # camera_info 내의 P(Projection) 행렬 유효성 간단 체크
            if msg_info1.p[0] == 0.0:
                self.get_logger().warn('CameraInfo의 P 행렬이 0입니다. 기선거리(Baseline) 연산에 오류가 있을 수 있습니다.', once=True)

            # 스테레오 매칭 연산 수행 (Disparity Map 계산)
            disparity = self.stereo.compute(cv_infra1, cv_infra2).astype(np.float32) / 16.0

            # 시차(Disparity) 데이터를 시각화 및 깊이 값으로 활용 가능한 8비트 이미지로 정규화
            # (거리가 유효하지 않은 영역은 0 처리)
            disparity[disparity <= 0] = 0
            depth_image = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # 복원된 이미지를 다시 ROS 2 이미지 메시지로 변환 (원본 타임스탬프 복사)
            msg_depth = self.bridge.cv2_to_imgmsg(depth_image, encoding='mono8')
            msg_depth.header = msg_infra1.header  # 시간 동기화를 위해 좌측 이미지 스탬프 그대로 주입

            # 토픽 발행
            self.pub_depth.publish(msg_depth)

        except Exception as e:
            self.get_logger().error(f'이미지 처리 중 오류 발생: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = InfraToDepthConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()