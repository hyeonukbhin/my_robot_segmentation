import pandas as pd
import matplotlib.pyplot as plt

# 1. CSV 데이터 불러오기
# 데이터 중 열 개수가 맞지 않게 꼬인 줄(bad lines)을 자동으로 건너뜁니다.
file_path = 'bhin_85.csv'
df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)

# 2. 데이터 전처리 및 에러 필터링
# 문자열(예: 'a') 등 비정상적인 데이터를 숫자로 변환하며, 변환 불가 시 NaN으로 처리합니다.
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
df['lin_vel'] = pd.to_numeric(df['lin_vel'], errors='coerce')
df['ang_vel'] = pd.to_numeric(df['ang_vel'], errors='coerce')

# 결측치(NaN)가 포함된 비정상 행을 완전히 제거합니다.
df = df.dropna()

# 타임스탬프를 주행 시작 기준(0초)의 상대 시간(초)으로 변환합니다.
df['time_sec'] = df['timestamp'] - df['timestamp'].iloc[0]

# 3. 그래프 그리기 (위: 선속도, 아래: 각속도)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- 선속도 (Linear Velocity) 그래프 ---
ax1.plot(df['time_sec'], df['lin_vel'], label='Linear Velocity', color='#1f77b4', linewidth=1.5)
ax1.set_title('Robot Linear Velocity Over Time')
ax1.set_ylabel('Velocity (m/s)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper right')

# --- 각속도 (Angular Velocity) 그래프 ---
ax2.plot(df['time_sec'], df['ang_vel'], label='Angular Velocity', color='#ff7f0e', linewidth=1.5)
ax2.set_title('Robot Angular Velocity Over Time')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='upper right')

# 레이아웃 간격 자동 조정
plt.tight_layout()

# 4. 이미지 저장 및 화면 출력
plt.savefig('robot_velocity_profile.png', dpi=300)
print("그래프가 'robot_velocity_profile.png' 이름으로 성공적으로 저장되었습니다.")
plt.show()