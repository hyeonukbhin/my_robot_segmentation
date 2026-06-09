#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# OpenVINO 엔진 로드 (같은 scripts 폴더 내에 있으므로 직접 import)
from inference_engine import SegformerEngineOV

class ExperimentNode:
    def __init__(self):
        # 1. ROS 1 노드 초기화
        rospy.init_node('experiment_segmentation_node', anonymous=False)
        self.bridge = CvBridge()
        
        rospy.loginfo("=== 구동 모드: ROS 1 OpenVINO Segformer (초경량화: Floor Mask Only) ===")

        # 2. 초경량화 엔진 초기화
        self.engine = SegformerEngineOV()

        # 3. 구독자 및 발행자 설정 (ROS 1 표준 문법, 딜레이 방지를 위해 queue_size=1)
        self.sub_rgb = rospy.Subscriber('/d456_camera/color/image_raw', Image, self.image_callback, queue_size=1)
        self.pub_floor = rospy.Publisher('/segmentation/floor_mask', Image, queue_size=1)

    def image_callback(self, rgb_msg):
        try:
            # 1. ROS 이미지 -> OpenCV 변환
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")

            # 2. 엔진 추론 (오직 Floor Mask 단일 반환)
            floor_mask = self.engine.infer(cv_rgb)

            # 3. 마스크 발행
            img_msg = self.bridge.cv2_to_imgmsg(floor_mask, "mono8")
            img_msg.header = rgb_msg.header # 원본 타임스탬프 및 프레임 ID 유지
            self.pub_floor.publish(img_msg)
            
        except Exception as e:
            rospy.logerr(f"이미지 처리 중 에러 발생: {e}")

if __name__ == '__main__':
    try:
        # 노드 실행 및 무한 대기
        node = ExperimentNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("노드가 정상적으로 종료되었습니다.")
