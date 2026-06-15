# check_realsense.py
import pyrealsense2 as rs

def check_realsense_config():
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color)
        profile = pipeline.start(config)

        color_sensor = profile.get_device().query_sensors()[1]
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        
        print("\n=== 📷 리얼센스 하드웨어 상태 점검 ===")
        print(f"현재 연결 해상도: {color_profile.width()} x {color_profile.height()}")
        print(f"펌웨어/드라이버 FPS 락: {color_profile.fps()} FPS")

        if color_sensor.supports(rs.option.enable_auto_exposure):
            ae = color_sensor.get_option(rs.option.enable_auto_exposure)
            print(f"Auto Exposure: {'ON (프레임 드랍 유발 가능)' if ae else 'OFF'}")
        
        if color_sensor.supports(rs.option.auto_exposure_priority):
            ae_priority = color_sensor.get_option(rs.option.auto_exposure_priority)
            print(f"저조도 FPS 하향 기능: {'ON (강제로 FPS가 떨어집니다)' if ae_priority else 'OFF'}")

        pipeline.stop()
        print("=======================================\n")
        
    except Exception as e:
        print(f"❌ 리얼센스 연결 실패: {e}")

if __name__ == "__main__":
    check_realsense_config()