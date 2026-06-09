import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2

from nav_msgs.msg import OccupancyGrid, MapMetaData
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from .inference_engine import SegformerEngine
from .inference_engine import NYURGBDEngine
from .depth_filter import DepthNoiseFilter
from .semantic_costmap import SemanticCostmapGenerator 
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

class ExperimentNode(Node):
    def __init__(self):
        super().__init__('experiment_segmentation_node')
        self.bridge = CvBridge()

        self.declare_parameter('mode', 1)
        self.mode = self.get_parameter('mode').value
        
        self.get_logger().info(f"=== 현재 실험 모드: {self.mode} ===")

        self.colors = {
            'Floor': (0, 255, 0), 'Wall': (128, 128, 128), 'Ceiling': (0, 0, 255), 
            'Door': (255, 165, 0), 'Window': (0, 255, 255), 'Glass': (200, 200, 0), 'Trash': (255, 0, 0)
        }

        # 🌟 상위 딕셔너리 관리: 여기서 주석 처리된 항목은 절대 검출되지 않습니다.
        if self.mode in [1, 2]:
            self.target_objects = {
                8: 'Window', 
                # 14: 'Door',  # 주석 처리 시 완벽히 무시됨
                138: 'Trash', 
                147: 'Glass'
            }
            self.engine = SegformerEngine(target_objects=self.target_objects)
        elif self.mode in [3, 4]:
            self.target_objects = {
                # 8: 'Door', 
                9: 'Window', 
                33: 'Trash'
            }
            weight_path = '/home/bhin/ros2_ws/src/my_robot_segmentation/weights/nyuv2_r34.pth'
            self.engine = NYURGBDEngine(weight_path=weight_path, target_objects=self.target_objects)

        self.depth_filter = DepthNoiseFilter(
            camera_height=0.56, camera_pitch_deg=0.0, fx=388.176, fy=387.717, 
            cx=323.138, cy=246.427, height_tolerance=0.15)
        
        self.costmap_gen = SemanticCostmapGenerator(
            resolution=0.05, width=6.0, height=6.0, fx=388.176, fy=387.717, 
            cx=323.138, cy=246.427, camera_height=0.56)

        self.sub_rgb = message_filters.Subscriber(self, Image, '/d456_camera/color/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/d456_camera/depth/image_rect_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)

        self.sub_sem = self.create_subscription(Image, '/sem', self.sem_callback, 10)

        self.pub_floor = self.create_publisher(Image, '/segmentation/floor_mask', 10)
        self.pub_wall = self.create_publisher(Image, '/segmentation/wall_mask', 10)
        self.pub_ceiling = self.create_publisher(Image, '/segmentation/ceiling_mask', 10)
        
        self.pub_object = self.create_publisher(Image, '/segmentation/object_mask', 10)
        self.pub_overlay = self.create_publisher(Image, '/segmentation/overlay_image', 10)
        
        qos_profile = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE)
        
        self.pub_costmap = self.create_publisher(OccupancyGrid, '/segmentation/local_costmap', qos_profile)
        self.pub_old_costmap = self.create_publisher(OccupancyGrid, '/old_segmentation/local_costmap', qos_profile) 
        
        self.pub_blind_spot = self.create_publisher(Marker, '/segmentation/blind_spot_marker', 10)

    def sync_callback(self, rgb_msg, depth_msg):
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1") 

        wall, floor, ceiling, _, obj_details = self.engine.infer(cv_rgb, cv_depth)

        if self.mode in [2, 4]:
            final_floor = self.depth_filter.refine_floor_mask(floor, cv_depth)
        else:
            final_floor = floor 

        header = rgb_msg.header
        
        self.pub_floor.publish(self.bridge.cv2_to_imgmsg(final_floor, "mono8", header=header))
        self.pub_wall.publish(self.bridge.cv2_to_imgmsg(wall, "mono8", header=header))
        self.pub_ceiling.publish(self.bridge.cv2_to_imgmsg(ceiling, "mono8", header=header))

        # 🌟 2번 수정: Object 마스크 (RGB 원본과 블렌딩 해제, 순수 검은 배경 사용)
        object_overlay, _ = self.draw_colored_masks(cv_rgb, obj_details, draw_labels=True)
        self.pub_object.publish(self.bridge.cv2_to_imgmsg(object_overlay, "rgb8", header=header))

        # 전체 오버레이 영상 (바닥, 벽 포함)
        all_masks = {'Floor': final_floor, 'Wall': wall, 'Ceiling': ceiling, **obj_details}
        full_overlay, detected = self.draw_colored_masks(cv_rgb, all_masks, draw_labels=True)
        full_blended = cv2.addWeighted(cv_rgb, 0.6, full_overlay, 0.4, 0)
        
        y_off = 30
        for l in detected:
            c = self.colors.get(l, (255, 255, 255))
            cv2.putText(full_blended, l, (full_blended.shape[1]-120, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
            cv2.putText(full_blended, l, (full_blended.shape[1]-120, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
            y_off += 30

        self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(full_blended, "rgb8", header=header))

        local_costmap_2d = self.costmap_gen.generate_costmap(final_floor)
        self.publish_costmap_msg(local_costmap_2d, header, self.pub_costmap)
        self.publish_blind_spot_marker(header)

    def sem_callback(self, msg):
        try:
            cv_sem = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # 1. 흑백 변환
            if len(cv_sem.shape) == 3:
                cv_sem = cv2.cvtColor(cv_sem, cv2.COLOR_BGR2GRAY)
            
            # 🌟 [안전장치 1] 투영 공식 오류 방지를 위한 해상도 강제 고정 (640x480)
            if cv_sem.shape[:2] != (480, 640):
                cv_sem = cv2.resize(cv_sem, (640, 480), interpolation=cv2.INTER_NEAREST)
            
            # 2. 바닥 추출 (회색 128 배경 무시)
            if np.max(cv_sem) <= 40:
                floor_mask = (cv_sem == 2).astype(np.uint8) * 255
            else: 
                _, floor_mask = cv2.threshold(cv_sem, 200, 255, cv2.THRESH_BINARY)
                
            # 🌟 [안전장치 2] Frame ID 강제 주입 (로스백의 /sem 토픽 헤더가 비어있으면 RViz가 거부함)
            header = msg.header
            if not header.frame_id:
                header.frame_id = 'd456_camera_color_optical_frame'
                
            old_costmap = self.costmap_gen.generate_costmap(floor_mask)
            self.publish_costmap_msg(old_costmap, header, self.pub_old_costmap)
            
        except Exception as e:
            self.get_logger().error(f'/sem 토픽 처리 오류: {e}')

    def draw_colored_masks(self, base_img, masks_dict, draw_labels=True):
        # 🌟 np.zeros_like를 통해 입력 이미지와 동일한 크기의 '완벽한 검은색 배경'을 자동 생성합니다.
        overlay = np.zeros_like(base_img)
        detected = []

        for label, mask in masks_dict.items():
            if mask is not None and cv2.countNonZero(mask) > 500:
                detected.append(label)
                color = self.colors.get(label, (255, 255, 255))
                overlay[mask == 255] = color
                
                if draw_labels and label not in ['Floor', 'Wall', 'Ceiling']:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) > 800:
                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                                cv2.putText(overlay, label, (cX-20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return overlay, detected

    def publish_costmap_msg(self, costmap_array, header, publisher):
        grid = OccupancyGrid()
        grid.header = header
        grid.info = MapMetaData()
        grid.info.resolution = float(self.costmap_gen.res)
        grid.info.width, grid.info.height = int(self.costmap_gen.grid_w), int(self.costmap_gen.grid_h)
        grid.info.origin.position.x = - (grid.info.width * grid.info.resolution) / 2.0
        grid.info.origin.position.y = 0.56
        grid.info.origin.orientation.x = 0.7071068
        grid.info.origin.orientation.w = 0.7071068
        grid.data = costmap_array.flatten().astype(np.int8).tolist()
        publisher.publish(grid)

    # 🌟 3번 수정: 정확한 U자 형태의 블라인드 스팟 (카메라 시야 외곽) 마커 생성
    def publish_blind_spot_marker(self, header):
        pts = self.costmap_gen.get_blind_spot_points()
        y, z, limit = pts['y'], pts['z_min'], pts['limit_x']
        
        # U자 형태를 구성하기 위한 핵심 8개 좌표
        p_bl = Point(x=-limit, y=y, z=0.0)             # 바닥 좌측 (로봇 베이스)
        p_br = Point(x=limit, y=y, z=0.0)              # 바닥 우측 (로봇 베이스)
        p_ml = Point(x=-limit, y=y, z=z)               # 좌측 뎁스 경계
        p_mr = Point(x=limit, y=y, z=z)                # 우측 뎁스 경계
        p_tl = Point(x=-limit, y=y, z=pts['z_hit_l'])  # 좌측 최대 투영 끝점
        p_tr = Point(x=limit, y=y, z=pts['z_hit_right']) # 우측 최대 투영 끝점
        p_fov_l = Point(x=pts['x_l'], y=y, z=z)        # 카메라 시야 좌측 한계선
        p_fov_r = Point(x=pts['x_r'], y=y, z=z)        # 카메라 시야 우측 한계선

        # 1. FOV 경계선 (V자 라인)
        line = Marker()
        line.header = header; line.ns = "blind_spot"; line.id = 0
        line.type = Marker.LINE_STRIP; line.action = Marker.ADD
        line.scale.x = 0.04
        line.color.r = 1.0; line.color.a = 1.0
        line.points = [p_tl, p_fov_l, p_fov_r, p_tr]

        # 2. U자형 블라인드 스팟 영역 (폴리곤)
        poly = Marker()
        poly.header = header; poly.ns = "blind_spot"; poly.id = 1
        poly.type = Marker.TRIANGLE_LIST; poly.action = Marker.ADD
        poly.scale.x = 1.0; poly.scale.y = 1.0; poly.scale.z = 1.0
        poly.color.r = 1.0; poly.color.g = 0.3; poly.color.b = 0.3; poly.color.a = 0.4 # 붉은색 반투명
        
        # 로봇 앞부터 z_min까지의 사각형 + FOV 바깥쪽 좌/우 삼각형 = 완벽한 U자 
        poly.points = [
            p_bl, p_br, p_ml,     # 사각형 1 (바닥)
            p_br, p_mr, p_ml,     # 사각형 2 (바닥)
            p_ml, p_fov_l, p_tl,  # 좌측 시야 밖 삼각형
            p_mr, p_tr, p_fov_r   # 우측 시야 밖 삼각형
        ]

        # 3. 텍스트 마커
        text = Marker()
        text.header = header; text.ns = "blind_spot"; text.id = 2
        text.type = Marker.TEXT_VIEW_FACING; text.action = Marker.ADD
        text.scale.z = 0.25
        text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 1.0
        text.pose.position.x = 0.0
        text.pose.position.y = y
        text.pose.position.z = z / 2.0 # 텍스트를 로봇 본체 앞 U자형의 중앙에 배치
        text.text = "Blind Spot"

        self.pub_blind_spot.publish(line)
        self.pub_blind_spot.publish(poly)
        self.pub_blind_spot.publish(text)

def main(args=None):
    rclpy.init(args=args)
    node = ExperimentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()