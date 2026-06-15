#!/usr/bin/env python3
import os
import sys
import time           # 🌟 시간 측정을 위해 추가
import numpy as np    # 🌟 배열 값 변경(Grayscale)을 위해 추가
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
        
        # 🌟 핵심: /sem 토픽의 그레이스케일 픽셀값을 ROS 파라미터로 받습니다. (기본값: 128)
        self.sem_grayscale_value = rospy.get_param('~sem_grayscale_value', 128)
        
        rospy.loginfo("=== 구동 모드: ROS 1 OpenVINO (Zero-Latency, Decoupling callback Adapted) ===")
        rospy.loginfo(f"=== 🎨 SEM 토픽 그레이스케일 설정값: {self.sem_grayscale_value} ===")

        self.engine = SegformerEngineOV()
        
        # 콜백과 메인 연산을 분리하기 위한 최신 메시지 보관 변수
        self.latest_msg = None

        self.sub_rgb = rospy.Subscriber(
            '/d456_camera/color/image_raw', 
            Image, 
            self.image_callback, 
            queue_size=1,
            buff_size=2**24,
            tcp_nodelay=True
        )
        
        self.pub_floor = rospy.Publisher('/segmentation/floor_mask', Image, queue_size=1)
        # 🌟 신규: /sem 토픽 퍼블리셔 추가
        self.pub_sem = rospy.Publisher('/segmentation/sem', Image, queue_size=1)

    def image_callback(self, rgb_msg):
        # 콜백 함수는 연산을 하지 않고 변수만 덮어씁니다.
        self.latest_msg = rgb_msg

    def run(self):
        rate = rospy.Rate(30) 
        
        while not rospy.is_shutdown():
            if self.latest_msg is not None:
                # ⏱️ 전체 파이프라인 타이머 시작
                start_total = time.perf_counter()
                
                msg_to_process = self.latest_msg
                self.latest_msg = None

                try:
                    # ---------------------------------------------------------
                    # [Step 1] Fetch: ROS 메시지 -> CV2 변환
                    # ---------------------------------------------------------
                    start_fetch = time.perf_counter()
                    cv_rgb = self.bridge.imgmsg_to_cv2(msg_to_process, "rgb8")
                    time_fetch = (time.perf_counter() - start_fetch) * 1000

                    # ---------------------------------------------------------
                    # [Step 2] Process: AI 추론 진행
                    # ---------------------------------------------------------
                    start_process = time.perf_counter()
                    floor_mask = self.engine.infer(cv_rgb)
                    time_process = (time.perf_counter() - start_process) * 1000

                    # ---------------------------------------------------------
                    # [Step 3] Publish: 결과 마스크들을 ROS로 송신
                    # ---------------------------------------------------------
                    start_pub = time.perf_counter()
                    
                    # 3-1. 기존 floor_mask (255 값) 퍼블리시
                    img_msg = self.bridge.cv2_to_imgmsg(floor_mask, "mono8")
                    img_msg.header = msg_to_process.header # 원본 타임스탬프 유지
                    self.pub_floor.publish(img_msg)
                    
                    # 3-2. 🌟 신규 /sem 토픽 처리 (255 -> 파라미터 값으로 변경)
                    # 바닥(0보다 큰 값)인 영역을 사용자가 지정한 sem_grayscale_value로 덮어씁니다.
                    sem_mask = np.where(floor_mask > 0, self.sem_grayscale_value, 0).astype(np.uint8)
                    sem_msg = self.bridge.cv2_to_imgmsg(sem_mask, "mono8")
                    sem_msg.header = msg_to_process.header
                    self.pub_sem.publish(sem_msg)
                    
                    time_pub = (time.perf_counter() - start_pub) * 1000
                    time_total = (time.perf_counter() - start_total) * 1000

                    # ---------------------------------------------------------
                    # [Step 4] 프로파일링 로그 출력
                    # ---------------------------------------------------------
                    rospy.loginfo(f"⏱️ [Total: {time_total:.1f}ms] | Fetch: {time_fetch:.1f}ms | Process: {time_process:.1f}ms | Publish: {time_pub:.1f}ms")
                    
                except Exception as e:
                    rospy.logerr(f"이미지 처리 중 에러 발생: {e}")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        node = ExperimentNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("노드가 정상적으로 종료되었습니다.")