#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

class DummyTFBridge(Node):
    def __init__(self):
        super().__init__('dummy_tf_bridge')
        
        # 1. Static TF 브로드캐스터 생성
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
        # 2. 초기화 시 단 한 번만 발행
        self.publish_static_tf()
        
        # 3. 노드가 정상적으로 돌고 있는지 확인하기 위한 로그 출력
        self.get_logger().info("=== Static TF 발행 시작 (d456_camera_link -> d456_camera_color_optical_frame) ===")
        self.get_logger().info("=== 종료하려면 Ctrl+C를 누르세요 ===")

    def publish_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'd456_camera_link'
        t.child_frame_id = 'd456_camera_color_optical_frame'
        
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        t.transform.rotation.x = -0.5
        t.transform.rotation.y = 0.5
        t.transform.rotation.z = -0.5
        t.transform.rotation.w = 0.5
        
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = DummyTFBridge()
    try:
        rclpy.spin(node) # 여기서 노드가 죽지 않고 무한 루프를 돌며 대기합니다.
    except KeyboardInterrupt:
        node.get_logger().info("종료 요청됨.")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

# 🌟 파이썬 스크립트로 직접 실행(python3 ~)할 때 반드시 필요한 진입점
if __name__ == '__main__':
    main()