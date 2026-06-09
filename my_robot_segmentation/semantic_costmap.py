import numpy as np

class SemanticCostmapGenerator:
    def __init__(self, resolution=0.05, width=6.0, height=6.0, fx=388.176, fy=387.717, cx=323.138, cy=246.427, camera_height=0.56):
        self.res = resolution
        self.grid_w = int(width / resolution)
        self.grid_h = int(height / resolution)
        
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.h = camera_height

        c_grid, r_grid = np.meshgrid(np.arange(self.grid_w), np.arange(self.grid_h))
        
        # Grid Cell의 중심점으로 좌표 정밀 보정 (+ 0.5)
        self.Z_map = np.maximum((r_grid + 0.5) * self.res, 0.01)
        
        # 정면 좌표계 기준 및 좌우 대칭 교정이 완료된 수식
        self.X_map = (c_grid + 0.5 - (self.grid_w / 2.0)) * self.res

        # 정확한 픽셀 매핑을 위해 수학적 내림(floor) 후 정수 변환
        self.U_map = np.floor((self.fx * (self.X_map / self.Z_map)) + self.cx).astype(np.int32)
        self.V_map = np.floor((self.fy * (self.h / self.Z_map)) + self.cy).astype(np.int32)

    def generate_costmap(self, floor_mask):
        """
        floor_mask: RGB 세그멘테이션 결과 (mono8, 255: 바닥, 0: 장애물/배경)
        """
        img_h, img_w = floor_mask.shape

        # 카메라 화면 안에 정상적으로 들어오는 영역 마스크
        valid_fov = (self.U_map >= 0) & (self.U_map < img_w) & (self.V_map >= 0) & (self.V_map < img_h)

        # 빈 코스트맵 초기화 (-1: 알 수 없음, 미탐색 구역)
        costmap = np.full((self.grid_h, self.grid_w), -1, dtype=np.int8)

        # 화면에 보이는 영역 전사 (Wrapping)
        valid_U = self.U_map[valid_fov]
        valid_V = self.V_map[valid_fov]
        
        sampled_pixels = floor_mask[valid_V, valid_U]
        costmap[valid_fov] = np.where(sampled_pixels == 255, 0, 100)

        return costmap

    def get_blind_spot_points(self, img_w=640, img_h=480):
        """카메라 파라미터를 기반으로 사각지대 폴리곤의 3D 좌표를 동적 계산"""
        z_min = (self.fy * self.h) / (img_h - self.cy)

        x_left_zmin = (0 - self.cx) * z_min / self.fx
        x_right_zmin = (img_w - self.cx) * z_min / self.fx

        costmap_half_w = (self.grid_w * self.res) / 2.0
        z_hit_left = (-costmap_half_w * self.fx) / (0 - self.cx)
        z_hit_right = (costmap_half_w * self.fx) / (img_w - self.cx)

        return {
            'y': self.h, 
            'z_min': z_min,
            'x_l': x_left_zmin,
            'x_r': x_right_zmin,
            'z_hit_l': z_hit_left,
            'z_hit_right': z_hit_right,
            'limit_x': costmap_half_w
        }