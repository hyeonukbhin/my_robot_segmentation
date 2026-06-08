import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import numpy as np

# ROS 2 내비게이션 표준 메시지 추가
from nav_msgs.msg import OccupancyGrid, MapMetaData
# 🌟 시각화 마커를 위한 메시지 추가
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from .inference_engine import SegformerEngine, NYURGBDEngine
from .depth_filter import DepthNoiseFilter
from .semantic_costmap import SemanticCostmapGenerator # 신규 모듈 임포트
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

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
            
        # 카메라 캘리브레이션 프로필 상수가 기입된 뎁스 노이즈 필터
        self.depth_filter = DepthNoiseFilter(
            camera_height=0.56,       
            camera_pitch_deg=0.0,    
            fx=388.176,              
            fy=387.717,              
            cx=323.138,              
            cy=246.427,              
            height_tolerance=0.15    
        )

        # 2D 로컬 코스트맵 생성기 초기화 (노이즈 필터와 파라미터 동기화)
        self.costmap_gen = SemanticCostmapGenerator(
            resolution=0.05,
            width=6.0,
            height=6.0,
            fx=388.176,
            fy=387.717,
            cx=323.138,
            cy=246.427
        )

        # 토픽 구독 및 시간 동기화
        self.sub_rgb = message_filters.Subscriber(self, Image, '/d456_camera/color/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/d456_camera/depth/recovered_image')

        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)

        # 토픽 발행자 등록
        self.pub_floor = self.create_publisher(Image, '/segmentation/floor_mask', 10)
        self.pub_wall = self.create_publisher(Image, '/segmentation/wall_mask', 10)
        
        # 2D 로컬 코스트맵 전용 퍼블리셔 등록 (QoS 프로필 적용)
        qos_profile = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.pub_costmap = self.create_publisher(OccupancyGrid, '/segmentation/local_costmap', qos_profile)
        
        # 🌟 사각지대 시각화 마커 퍼블리셔 추가
        self.pub_blind_spot = self.create_publisher(Marker, '/segmentation/blind_spot_marker', 10)

    def sync_callback(self, rgb_msg, depth_msg):
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1") 

        # 1. 모델 추론
        wall_mask, floor_mask = self.engine.infer(cv_rgb, cv_depth)

        # 2. (선택적) 뎁스 필터링 
        if self.mode in [2, 4]:
            final_floor_mask = self.depth_filter.refine_floor_mask(floor_mask, cv_depth)
        else:
            final_floor_mask = floor_mask 

        # 3. 기존 이미지 토픽 발행 (floor가 잘 오는지 RViz로 확인하기 위함)
        if final_floor_mask is not None:
            floor_msg = self.bridge.cv2_to_imgmsg(final_floor_mask, "mono8")
            floor_msg.header = rgb_msg.header
            self.pub_floor.publish(floor_msg)

        # 4. RGB 전용 코스트맵 생성 및 발행
        local_costmap_2d = self.costmap_gen.generate_costmap(final_floor_mask)
        self.publish_costmap_msg(local_costmap_2d, rgb_msg.header)
        
        # 🌟 5. 사각지대(Blind Spot) 마커 발행
        self.publish_blind_spot_marker(rgb_msg.header)

    # 🌟 원본에서 잘려있던 코스트맵 발행 함수 복구
    def publish_costmap_msg(self, costmap_array, header):
        grid_msg = OccupancyGrid()
        grid_msg.header = header
        
        grid_msg.info = MapMetaData()
        grid_msg.info.resolution = float(self.costmap_gen.res)
        grid_msg.info.width = int(self.costmap_gen.grid_w)
        grid_msg.info.height = int(self.costmap_gen.grid_h)
        
        grid_msg.info.origin.position.x = - (self.costmap_gen.grid_w * self.costmap_gen.res) / 2.0
        grid_msg.info.origin.position.y = 0.56  
        grid_msg.info.origin.position.z = 0.0
        
        # 🌟 변경: x의 부호를 양수(+)로 변경하여 맵을 정면으로 눕힙니다.
        grid_msg.info.origin.orientation.x = 0.7071068   # (기존 -0.7071068 에서 마이너스 제거)
        grid_msg.info.origin.orientation.y = 0.0
        grid_msg.info.origin.orientation.z = 0.0
        grid_msg.info.origin.orientation.w = 0.7071068

        grid_msg.data = costmap_array.flatten().astype(np.int8).tolist()
        self.pub_costmap.publish(grid_msg)

    # 🌟 사각지대 그래픽 마커 발행 함수
    def publish_blind_spot_marker(self, header):
        # 생성기에서 카메라 파라미터 기반 동적 좌표 가져오기 (640x480 기준)
        pts = self.costmap_gen.get_blind_spot_points(img_w=640, img_h=480)
        
        y = pts['y']
        limit_x = pts['limit_x']
        z_min = pts['z_min']
        x_l, x_r = pts['x_l'], pts['x_r']
        z_hit_l, z_hit_r = pts['z_hit_l'], pts['z_hit_right']

        # 1. 빨간색 테두리 선
        line_marker = Marker()
        line_marker.header = header
        line_marker.ns = "blind_spot_border"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.04 # 선 굵기
        line_marker.color.r = 1.0
        line_marker.color.a = 1.0  # 투명도 100%

        p1 = Point(x=-limit_x, y=y, z=z_hit_l)
        p2 = Point(x=x_l, y=y, z=z_min)
        p3 = Point(x=x_r, y=y, z=z_min)
        p4 = Point(x=limit_x, y=y, z=z_hit_r)
        line_marker.points = [p1, p2, p3, p4]

        # 2. 반투명한 면 채우기
        poly_marker = Marker()
        poly_marker.header = header
        poly_marker.ns = "blind_spot_fill"
        poly_marker.id = 1
        poly_marker.type = Marker.TRIANGLE_LIST
        poly_marker.action = Marker.ADD
        poly_marker.scale.x = 1.0; poly_marker.scale.y = 1.0; poly_marker.scale.z = 1.0
        
        # 연한 노란색/미색
        poly_marker.color.r = 1.0; poly_marker.color.g = 0.95; poly_marker.color.b = 0.8
        poly_marker.color.a = 0.4 # 반투명
        
        p_bl = Point(x=-limit_x, y=y, z=0.0)
        p_br = Point(x=limit_x, y=y, z=0.0)

        poly_marker.points.extend([p1, p2, p_bl])
        poly_marker.points.extend([p2, p3, p_bl])
        poly_marker.points.extend([p_bl, p3, p_br])
        poly_marker.points.extend([p3, p4, p_br])

        # 3. "Blind Spot" 텍스트 마커
        text_marker = Marker()
        text_marker.header = header
        text_marker.ns = "blind_spot_text"
        text_marker.id = 2
        text_marker.type = Marker.TEXT_VIEW_FACING 
        text_marker.action = Marker.ADD
        
        text_marker.text = "Blind Spot"
        
        # 텍스트 위치: 사각지대 정중앙
        text_marker.pose.position.x = 0.0
        text_marker.pose.position.y = y
        text_marker.pose.position.z = z_min / 2.0 
        
        text_marker.scale.z = 0.5 # 텍스트 크기
        text_marker.color.r = 1.0 # 텍스트 색상
        text_marker.color.a = 1.0 

        self.pub_blind_spot.publish(line_marker)
        self.pub_blind_spot.publish(poly_marker)
        self.pub_blind_spot.publish(text_marker)


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()