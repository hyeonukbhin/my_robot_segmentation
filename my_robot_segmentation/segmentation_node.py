import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# OpenVINO 엔진 로드
from .inference_engine import SegformerEngineOV

class ExperimentNode(Node):
    def __init__(self):
        super().__init__('experiment_segmentation_node')
        self.bridge = CvBridge()
        
        self.get_logger().info("=== 구동 모드: OpenVINO Segformer (초경량화: Floor Mask Only) ===")

        # 🌟 초경량화 엔진 초기화
        self.engine = SegformerEngineOV()

        # 🌟 오직 RGB 구독 및 Floor 마스크 발행만 남김
        self.sub_rgb = self.create_subscription(Image, '/d456_camera/color/image_raw', self.image_callback, 10)
        self.pub_floor = self.create_publisher(Image, '/segmentation/floor_mask', 10)

        # =====================================================================
        # 아래 기능들은 연산량 감소를 위해 전부 비활성화(주석 처리) 하였습니다.
        # =====================================================================
        # from .semantic_costmap import SemanticCostmapGenerator 
        # from nav_msgs.msg import OccupancyGrid
        # from visualization_msgs.msg import Marker
        # self.costmap_gen = SemanticCostmapGenerator(...)
        # self.sub_sem = self.create_subscription(Image, '/sem', self.sem_callback, 10)
        # self.pub_wall = self.create_publisher(Image, '/segmentation/wall_mask', 10)
        # self.pub_ceiling = self.create_publisher(Image, '/segmentation/ceiling_mask', 10)
        # self.pub_object = self.create_publisher(Image, '/segmentation/object_mask', 10)
        # self.pub_overlay = self.create_publisher(Image, '/segmentation/overlay_image', 10)
        # self.pub_costmap = self.create_publisher(OccupancyGrid, '/segmentation/local_costmap', qos_profile)
        # self.pub_blind_spot = self.create_publisher(Marker, '/segmentation/blind_spot_marker', 10)

    def image_callback(self, rgb_msg):
        # 1. ROS 이미지 -> OpenCV 변환
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")

        # 2. 엔진 추론 (오직 Floor Mask 단일 반환)
        floor_mask = self.engine.infer(cv_rgb)

        # 3. 마스크 발행
        header = rgb_msg.header
        self.pub_floor.publish(self.bridge.cv2_to_imgmsg(floor_mask, "mono8", header=header))

        # =====================================================================
        # 오버레이 이미지 생성 및 코스트맵 투영 연산 제거 (CPU 점유율 대폭 하락)
        # =====================================================================

def main(args=None):
    rclpy.init(args=args)
    node = ExperimentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()