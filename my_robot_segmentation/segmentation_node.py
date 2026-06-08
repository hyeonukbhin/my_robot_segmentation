import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

from .inference_engine import SegformerEngine, NYURGBDEngine
from .depth_filter import DepthNoiseFilter

class ExperimentNode(Node):
    def __init__(self):
        super().__init__('experiment_segmentation_node')
        self.bridge = CvBridge()
        
        # 실험 모드 파라미터 선언 (기본값: 1)
        self.declare_parameter('mode', 1)
        self.mode = self.get_parameter('mode').value
        self.get_logger().info(f"=== 현재 실험 모드: {self.mode} ===")

        # 모드에 따른 엔진 초기화
        if self.mode in [1, 2]:
            self.get_logger().info("Hugging Face RGB 모델(SegFormer)을 로드합니다.")
            self.engine = SegformerEngine()
        elif self.mode in [3, 4]:
            self.get_logger().info("NYUv2 RGB-D 공식 모델(ESANet)을 로드합니다.")
            weight_path = "/home/bhin/ros2_ws/src/my_robot_segmentation/weights/nyuv2_r34.pth"
            self.engine = NYURGBDEngine(weight_path)
            
        self.depth_filter = DepthNoiseFilter(
            camera_height=0.56,       # ⚠️ [수정 필수] 줄자로 실제 로봇 바닥에서 카메라 렌즈 중심까지의 높이(m)를 재서 입력해 주세요. (예: 50cm면 0.5)
            camera_pitch_deg=0.0,    # 지면과 완벽히 수평하므로 0도
            fx=388.176,              # k 행렬 1번째 값
            fy=387.717,              # k 행렬 5번째 값
            cx=323.138,              # k 행렬 3번째 값
            cy=246.427,              # k 행렬 6번째 값
            height_tolerance=0.15    # 지면 오차 허용 범위 (±5cm)
        )
        # 토픽 구독 및 시간 동기화
        #self.sub_rgb = message_filters.Subscriber(self, Image, '/d456_camera/color/image_raw')
        #self.sub_depth = message_filters.Subscriber(self, Image, '/d456_camera/depth/recovered_image')

        self.sub_rgb = message_filters.Subscriber(self, Image, '/d400/color/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/d400/depth/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)

        # 토픽 발행자 등록 (바닥 및 벽 퍼블리셔 모두 생성)
        self.pub_floor = self.create_publisher(Image, '/segmentation/floor_mask', 10)
        self.pub_wall = self.create_publisher(Image, '/segmentation/wall_mask', 10)

    def sync_callback(self, rgb_msg, depth_msg):
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1") 

        # 1. 모델 추론 (공통)
        # inference_engine에서 벽과 바닥 마스크를 분리하여 받아옴
        wall_mask, floor_mask = self.engine.infer(cv_rgb, cv_depth)

        # 2. 후처리: 바닥 마스크에 대한 뎁스 필터링 적용 여부 결정
        # 모드 2 (RGB + 필터링) 또는 모드 4 (RGB-D + 필터링)일 때만 작동
        if self.mode == 2 or self.mode == 4:
            final_floor_mask = self.depth_filter.refine_floor_mask(floor_mask, cv_depth)
        else:
            final_floor_mask = floor_mask # 필터링 없이 그대로 출력 (모드 1, 3)

        # 3. 토픽 발행 (바닥 및 벽 마스크 각각 독립적으로 전송)
        # 3-1. 바닥 마스크 발행
        floor_msg = self.bridge.cv2_to_imgmsg(final_floor_mask, "mono8")
        floor_msg.header = rgb_msg.header
        self.pub_floor.publish(floor_msg)

        # 3-2. 벽 마스크 발행 (추가된 부분)
        wall_msg = self.bridge.cv2_to_imgmsg(wall_mask, "mono8")
        wall_msg.header = rgb_msg.header
        self.pub_wall.publish(wall_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ExperimentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()