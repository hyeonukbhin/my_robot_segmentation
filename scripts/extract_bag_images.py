#!/usr/bin/env python3

import os
import cv2
import random
import rosbag
from cv_bridge import CvBridge, CvBridgeError

def extract_random_images_from_bag(bag_path, topic_name, output_dir, target_samples):
    """
    ROS Bag 파일에서 특정 이미지 토픽을 읽어 랜덤하게 N장을 추출합니다.
    """
    # 1. 출력 디렉토리 확인 및 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 출력 폴더 생성 완료: {output_dir}")

    bridge = CvBridge()
    
    print(f"📦 Bag 파일 분석 중: {bag_path}")
    bag = rosbag.Bag(bag_path, 'r')

    # 2. 토픽 정보 및 전체 메시지 개수(O(1) 속도) 확인
    info = bag.get_type_and_topic_info()
    if topic_name not in info.topics:
        print(f"❌ 오류: 토픽 '{topic_name}'을 Bag 파일에서 찾을 수 없습니다.")
        bag.close()
        return

    total_msgs = info.topics[topic_name].message_count
    print(f"📊 '{topic_name}' 토픽의 총 이미지 수: {total_msgs}장")

    # 추출할 개수가 전체 개수보다 많으면 전체 추출로 조정
    extract_count = min(target_samples, total_msgs)
    
    # 3. 전체 인덱스 중에서 랜덤하게 목표 개수만큼 뽑아 오름차순 정렬
    # (메모리에 이미지를 다 올리지 않고, 뽑을 순서만 미리 정해두는 핵심 로직)
    random_indices = set(random.sample(range(total_msgs), extract_count))
    print(f"🎲 랜덤 추출 타겟 수: {extract_count}장 (메시지 순회 시작...)")

    current_idx = 0
    saved_count = 0

    # 4. Bag 파일 순회하며 해당 인덱스의 이미지만 추출
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if current_idx in random_indices:
            try:
                # 데이터 타입에 따라 자동으로 cv2 변환 처리
                if msg._type == 'sensor_msgs/Image':
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                elif msg._type == 'sensor_msgs/CompressedImage':
                    cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                else:
                    print(f"⚠️ 지원하지 않는 메시지 타입입니다: {msg._type}")
                    continue
                
                # 파일명 지정 및 저장 (타임스탬프 기반)
                filename = os.path.join(output_dir, f"frame_{t.to_nsec()}.jpg")
                cv2.imwrite(filename, cv_img)
                saved_count += 1
                
                # 진행률 출력 (10장 단위)
                if saved_count % 10 == 0 or saved_count == extract_count:
                    print(f"⏳ 진행률: {saved_count}/{extract_count} 장 저장 완료...")

            except CvBridgeError as e:
                print(f"❌ 변환 에러 발생: {e}")
                
        current_idx += 1
        
        # 목표 수량을 다 채웠으면 순회 조기 종료
        if saved_count >= extract_count:
            break

    bag.close()
    print(f"✅ 추출 완료! 총 {saved_count}장의 랜덤 이미지가 '{output_dir}'에 저장되었습니다.")

if __name__ == '__main__':
    # ==========================================
    # 💡 여기서 본인의 환경에 맞게 경로를 수정하세요!
    # ==========================================
    BAG_FILE_PATH = "/external_data/amr_rgbd_ros1_260608.bag"  # ROS Bag 파일 절대/상대 경로
    IMAGE_TOPIC = "/d456_camera/color/image_raw"                    # 추출할 카메라 토픽 이름
    OUTPUT_FOLDER = "/external_data/data/calibration"                      # 이미지를 저장할 폴더
    NUM_SAMPLES = 150                                          # 양자화용으로 뽑을 랜덤 장수 (100~200 추천)

    extract_random_images_from_bag(BAG_FILE_PATH, IMAGE_TOPIC, OUTPUT_FOLDER, NUM_SAMPLES)