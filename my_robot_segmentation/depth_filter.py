import numpy as np

class DepthNoiseFilter:
    def __init__(self, camera_height=0.5, camera_pitch_deg=0.0, 
                 fx=388.176, fy=387.717, cx=323.138, cy=246.427,
                 height_tolerance=0.05):
        """
        camera_height: 로봇 바닥(지면)에서 카메라 중심까지의 물리적 높이 (m)
        camera_pitch_deg: 카메라가 아래를 향한 피치 각도 (degrees)
        fx, fy, cx, cy: 카메라 내재 변수 (Intrinsic Parameters)
        height_tolerance: 지면으로 인정할 높이 오차 범위 (m)
        """
        self.h = camera_height
        self.theta = np.radians(camera_pitch_deg)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.tolerance = height_tolerance

    def refine_floor_mask(self, floor_mask, cv_depth):
        h, w = cv_depth.shape
        refined_mask = floor_mask.copy()

        # 1. 픽셀 좌표 그리드 생성 (u, v)
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

        # 2. 유효한 뎁스 데이터 필터링 (0 또는 NaN 제거)
        valid_depth_mask = (cv_depth > 0) & (~np.isnan(cv_depth))

        # 3. 내재 변수를 이용한 카메라 좌표계 3D 공간 복원 (y_cam, z_cam 계산)
        # 바닥 높이(Z_base) 계산에는 x_cam이 필요 없으므로 연산 최적화를 위해 생략
        z_cam = np.where(valid_depth_mask, cv_depth, 0.0)
        y_cam = np.where(valid_depth_mask, (v_coords - self.cy) * z_cam / self.fy, 0.0)

        # 4. 전사 공식을 활용해 로봇 베이스 기준의 3D 높이(Z_base) 계산
        z_base = -z_cam * np.sin(self.theta) + y_cam * np.cos(self.theta) + self.h

        # [수정된 5번 로직] 유연한 높이 기반 필터링 적용
        # 뎁스 데이터가 존재하는(valid) 픽셀 중에서, 지면 높이를 벗어난 픽셀만 장애물로 간주해 지웁니다.
        # 반짝이는 바닥 때문에 뎁스가 0으로 뚫린 곳은 지우지 않고 AI의(RGB) 판단을 그대로 믿습니다.
        
        invalid_height = (np.abs(z_base) > self.tolerance) & valid_depth_mask
        
        refined_mask[invalid_height] = 0

        return refined_mask