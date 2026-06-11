#!/usr/bin/env python3
import os
import sys
import rospy
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('my_robot_segmentation')
sys.path.append(os.path.join(pkg_path, 'scripts'))

from inference_engine import SegformerEngineOV

class ExperimentNode:
    def __init__(self):
        rospy.init_node('experiment_segmentation_node', anonymous=False)
        self.bridge = CvBridge()
        
        rospy.loginfo("=== 구동 모드: ROS 1 OpenVINO (Zero-Latency, Decoupling callback Adapted) ===")

        self.engine = SegformerEngineOV()
        
        # 🌟 핵심: 콜백과 메인 연산을 분리하기 위한 최신 메시지 보관 변수
        self.latest_msg = None

        self.sub_rgb = rospy.Subscriber(
            '/d456_camera/color/image_raw', 
            Image, 
            self.image_callback, 
            queue_size=1,
            buff_size=2**24,      # 버퍼를 넉넉히 주어 네트워크 병목 방지
            tcp_nodelay=True
        )
        
        self.pub_floor = rospy.Publisher('/segmentation/floor_mask', Image, queue_size=1)

    def image_callback(self, rgb_msg):
        # 🌟 콜백 함수는 연산을 하지 않습니다! 
        # 그냥 최신 이미지 변수를 덮어쓰고 빛의 속도로 종료됩니다. (버퍼에 쌓일 틈이 없음)
        self.latest_msg = rgb_msg

    def run(self):
        # 초당 30번 확인 (필요에 따라 조절 가능, 엔진이 17FPS면 충분히 여유로움)
        rate = rospy.Rate(30) 
        
        while not rospy.is_shutdown():
            if self.latest_msg is not None:
                # 1. 최신 이미지를 가져오고, 변수를 비움 (중복 처리 방지)
                msg_to_process = self.latest_msg
                self.latest_msg = None

                # 2. 가져온 '가장 최신' 이미지로만 연산 진행
                try:
                    cv_rgb = self.bridge.imgmsg_to_cv2(msg_to_process, "rgb8")
                    floor_mask = self.engine.infer(cv_rgb)

                    img_msg = self.bridge.cv2_to_imgmsg(floor_mask, "mono8")
                    img_msg.header = msg_to_process.header # 원본 타임스탬프 유지
                    self.pub_floor.publish(img_msg)
                    
                except Exception as e:
                    rospy.logerr(f"이미지 처리 중 에러 발생: {e}")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        node = ExperimentNode()
        # rospy.spin() 대신 우리가 만든 비동기 메인 루프를 실행합니다.
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("노드가 정상적으로 종료되었습니다.")